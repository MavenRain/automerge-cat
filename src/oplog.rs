//! `OpLog`: operation-based replication for the document model.
//!
//! The state-based document model (`Document`) merges by joining
//! entire states.  The `OpLog` adds the operation-based (`CmRDT`)
//! perspective: each edit is a first-class value with causal
//! metadata, and replicas sync by exchanging operations.
//!
//! ## Categorical connection
//!
//! The causal history forms a directed acyclic graph (DAG) where:
//! - **Vertices** = document states (snapshots after each operation)
//! - **Edges** = operations (mutations that transition between states)
//!
//! The free category over this DAG (`comp_cat_rs::collapse::free_category`)
//! has paths as morphisms: a sequence of operations is a composite
//! edit.  The `interpret` function from the free category is exactly
//! the "apply" step: it maps each operation (edge) to a document
//! mutation (morphism in the category of document states), then
//! folds the path into a composite mutation.
//!
//! The `OpLog` itself is a join-semilattice: the join is the union of
//! operations and the union of causal edges.  This is a grow-only
//! structure, just like the CRDTs it builds on.

use std::collections::{BTreeMap, BTreeSet};

use comp_cat_rs::foundation::JoinSemilattice;
use serde::{Deserialize, Serialize};
use crate::document::{Document, NodeId, Value};
use crate::error::Error;
use crate::replica::{ReplicaId, Tag, Timestamp};
use crate::rga::Origin;

// ---------------------------------------------------------------------------
// Op: a single document mutation
// ---------------------------------------------------------------------------

/// A document operation: a single atomic edit.
///
/// Each operation is identified by its [`Tag`] and records the
/// mutation it performs.  In the free category over the causal
/// graph, each `Op` is an edge (a generating morphism).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Op {
    tag: Tag,
    action: Action,
}

impl Op {
    /// The unique tag identifying this operation.
    #[must_use]
    pub fn tag(&self) -> Tag {
        self.tag
    }

    /// The action this operation performs.
    #[must_use]
    pub fn action(&self) -> &Action {
        &self.action
    }
}

/// The mutation an operation performs.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Action {
    /// Set a key in a map node.
    SetKey {
        /// The map node to modify.
        node: NodeId,
        /// The key to set.
        key: String,
        /// The value to write.
        value: Value,
    },
    /// Delete a key from a map node.
    DeleteKey {
        /// The map node to modify.
        node: NodeId,
        /// The key to delete.
        key: String,
    },
    /// Insert a value into a list node.
    ListInsert {
        /// The list node to modify.
        node: NodeId,
        /// The position to insert after.
        origin: Origin,
        /// The value to insert.
        value: Value,
    },
    /// Delete an element from a list node.
    ListDelete {
        /// The list node to modify.
        node: NodeId,
        /// The tag of the element to delete.
        element_tag: Tag,
    },
    /// Create a new empty map container.
    CreateMap,
    /// Create a new empty list container.
    CreateList,
}

// ---------------------------------------------------------------------------
// OpLog
// ---------------------------------------------------------------------------

/// An append-only log of operations with causal metadata.
///
/// The `OpLog` is a join-semilattice (grow-only operations + grow-only
/// causal edges).  It can be materialized into a [`Document`] by
/// applying all operations in causal order.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpLog {
    /// All operations, keyed by their unique tag.
    ops: BTreeMap<Tag, Op>,
    /// Causal dependencies: `deps[tag]` is the set of tags that
    /// `tag` causally depends on (its "parents" in the DAG).
    deps: BTreeMap<Tag, BTreeSet<Tag>>,
    /// The current "heads": operations with no dependents yet.
    /// These represent the frontier of the causal graph.
    heads: BTreeSet<Tag>,
}

impl OpLog {
    /// An empty log with no operations.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            ops: BTreeMap::new(),
            deps: BTreeMap::new(),
            heads: BTreeSet::new(),
        }
    }

    /// Append an operation that depends on the current heads.
    ///
    /// # Errors
    ///
    /// Returns [`Error::DuplicateOp`] if the tag already exists.
    pub fn append(
        &self,
        action: Action,
        replica: ReplicaId,
        timestamp: Timestamp,
    ) -> Result<Self, Error> {
        let tag = Tag::new(replica, timestamp);
        if self.ops.contains_key(&tag) {
            Err(Error::DuplicateOp { tag })
        } else {
            let op = Op { tag, action };
            let op_deps = self.heads.clone();
            let ops: BTreeMap<Tag, Op> = self
                .ops
                .iter()
                .map(|(k, v)| (*k, v.clone()))
                .chain(core::iter::once((tag, op)))
                .collect();
            let deps: BTreeMap<Tag, BTreeSet<Tag>> = self
                .deps
                .iter()
                .map(|(k, v)| (*k, v.clone()))
                .chain(core::iter::once((tag, op_deps)))
                .collect();
            let heads = core::iter::once(tag).collect();
            Ok(Self { ops, deps, heads })
        }
    }

    /// Append an operation with explicit dependencies (for remote ops).
    ///
    /// Use this when ingesting operations from another replica,
    /// where the dependencies are the remote operation's parents
    /// rather than the local heads.
    ///
    /// # Errors
    ///
    /// Returns [`Error::DuplicateOp`] if the tag already exists.
    pub fn append_with_deps(
        &self,
        action: Action,
        replica: ReplicaId,
        timestamp: Timestamp,
        op_deps: BTreeSet<Tag>,
    ) -> Result<Self, Error> {
        let tag = Tag::new(replica, timestamp);
        if self.ops.contains_key(&tag) {
            Err(Error::DuplicateOp { tag })
        } else {
            let op = Op { tag, action };
            let ops: BTreeMap<Tag, Op> = self
                .ops
                .iter()
                .map(|(k, v)| (*k, v.clone()))
                .chain(core::iter::once((tag, op)))
                .collect();
            let deps: BTreeMap<Tag, BTreeSet<Tag>> = self
                .deps
                .iter()
                .map(|(k, v)| (*k, v.clone()))
                .chain(core::iter::once((tag, op_deps)))
                .collect();
            let heads = Self::recompute_heads(&ops, &deps);
            Ok(Self { ops, deps, heads })
        }
    }

    /// The current heads (frontier of the causal graph).
    #[must_use]
    pub fn heads(&self) -> &BTreeSet<Tag> {
        &self.heads
    }

    /// All operation tags in the log.
    #[must_use]
    pub fn tags(&self) -> BTreeSet<Tag> {
        self.ops.keys().copied().collect()
    }

    /// The number of operations in the log.
    #[must_use]
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    /// Whether the log has no operations.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// Remove operations that are in the causal past of
    /// `frontier` and have been observed by all replicas.
    ///
    /// Operations in `safe_tags` are removed along with their
    /// dependency entries.  The remaining ops have their deps
    /// filtered to exclude removed tags.
    #[must_use]
    pub fn compact(&self, safe_tags: &BTreeSet<Tag>) -> Self {
        let ops: BTreeMap<Tag, Op> = self
            .ops
            .iter()
            .filter(|(tag, _)| !safe_tags.contains(tag))
            .map(|(k, v)| (*k, v.clone()))
            .collect();
        let deps: BTreeMap<Tag, BTreeSet<Tag>> = self
            .deps
            .iter()
            .filter(|(tag, _)| !safe_tags.contains(tag))
            .map(|(tag, d)| {
                let filtered: BTreeSet<Tag> = d
                    .iter()
                    .filter(|t| !safe_tags.contains(t))
                    .copied()
                    .collect();
                (*tag, filtered)
            })
            .collect();
        let heads = Self::recompute_heads(&ops, &deps);
        Self { ops, deps, heads }
    }

    /// Look up an operation by tag.
    #[must_use]
    pub fn get(&self, tag: Tag) -> Option<&Op> {
        self.ops.get(&tag)
    }

    /// The causal dependencies of an operation.
    #[must_use]
    pub fn dependencies(&self, tag: Tag) -> BTreeSet<Tag> {
        self.deps
            .get(&tag)
            .cloned()
            .unwrap_or_default()
    }

    /// Materialize the log into a [`Document`].
    ///
    /// For each operation, the predecessor state is the join of the
    /// states produced by its causal dependencies.  This ensures
    /// concurrent operations see the same base state (not each
    /// other's effects), giving correct `MvRegister` semantics.
    ///
    /// This is the free-category interpretation: each operation
    /// (edge) is mapped to a document mutation (morphism), and the
    /// final document is the join of all head states.
    #[must_use]
    pub fn materialize(&self) -> Document {
        let states = self
            .topo_sort()
            .iter()
            .fold(BTreeMap::<Tag, Document>::new(), |memo, tag| {
                let dep_tags = self.deps.get(tag).cloned().unwrap_or_default();
                let base = dep_tags
                    .iter()
                    .filter_map(|d| memo.get(d))
                    .fold(Document::new(), |acc, d| acc.join(d));
                let state = self
                    .ops
                    .get(tag)
                    .map_or_else(|| base.clone(), |o| apply_op(&base, o));
                memo.into_iter()
                    .chain(core::iter::once((*tag, state)))
                    .collect()
            });
        self.heads
            .iter()
            .filter_map(|h| states.get(h))
            .fold(Document::new(), |acc, d| acc.join(d))
    }

    /// Operations that are in `self` but not in `other`.
    ///
    /// Useful for incremental sync: send only the ops the remote
    /// replica is missing.
    #[must_use]
    pub fn diff(&self, other: &OpLog) -> Vec<&Op> {
        self.ops
            .iter()
            .filter(|(tag, _)| !other.ops.contains_key(tag))
            .map(|(_, op)| op)
            .collect()
    }

    /// Topological sort of operations respecting causal order.
    ///
    /// Operations with no dependencies come first; an operation
    /// always appears after all its dependencies.
    fn topo_sort(&self) -> Vec<Tag> {
        // Kahn's algorithm, functional style.
        // in_degree: how many unprocessed dependencies each op has.
        let in_degree: BTreeMap<Tag, usize> = self
            .ops
            .keys()
            .map(|tag| {
                let deg = self
                    .deps
                    .get(tag)
                    .map_or(0, |d| d.iter().filter(|t| self.ops.contains_key(t)).count());
                (*tag, deg)
            })
            .collect();

        let dependents: BTreeMap<Tag, Vec<Tag>> = self.ops.keys().fold(
            BTreeMap::new(),
            |acc, tag| {
                self.deps
                    .get(tag)
                    .map_or(acc.clone(), |d| {
                        d.iter()
                            .filter(|dep| self.ops.contains_key(dep))
                            .fold(acc, |inner, dep| {
                                let entry: Vec<Tag> = inner
                                    .get(dep)
                                    .cloned()
                                    .unwrap_or_default()
                                    .into_iter()
                                    .chain(core::iter::once(*tag))
                                    .collect();
                                inner
                                    .into_iter()
                                    .filter(|(k, _)| k != dep)
                                    .chain(core::iter::once((*dep, entry)))
                                    .collect()
                            })
                    })
            },
        );

        Self::topo_step(&in_degree, &dependents, Vec::new())
    }

    /// Recursive step of Kahn's algorithm.
    fn topo_step(
        in_degree: &BTreeMap<Tag, usize>,
        dependents: &BTreeMap<Tag, Vec<Tag>>,
        sorted: Vec<Tag>,
    ) -> Vec<Tag> {
        let ready: Vec<Tag> = in_degree
            .iter()
            .filter(|(_, deg)| **deg == 0)
            .map(|(tag, _)| *tag)
            .collect();

        if ready.is_empty() {
            sorted
        } else {
            let new_sorted: Vec<Tag> = sorted
                .into_iter()
                .chain(ready.iter().copied())
                .collect();

            let updated_in_degree: BTreeMap<Tag, usize> = in_degree
                .iter()
                .filter(|(tag, _)| !ready.contains(tag))
                .map(|(tag, deg)| {
                    let decrement = ready
                        .iter()
                        .filter(|r| {
                            dependents
                                .get(r)
                                .is_some_and(|ds| ds.contains(tag))
                        })
                        .count();
                    (*tag, deg.saturating_sub(decrement))
                })
                .collect();

            Self::topo_step(&updated_in_degree, dependents, new_sorted)
        }
    }

    /// Recompute heads: tags with no dependents.
    fn recompute_heads(
        ops: &BTreeMap<Tag, Op>,
        deps: &BTreeMap<Tag, BTreeSet<Tag>>,
    ) -> BTreeSet<Tag> {
        let all_deps: BTreeSet<Tag> = deps.values().flat_map(|d| d.iter().copied()).collect();
        ops.keys()
            .filter(|tag| !all_deps.contains(tag))
            .copied()
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Apply: interpret an operation as a document mutation
// ---------------------------------------------------------------------------

/// Apply a single operation to a document, producing the next state.
///
/// This is the `GraphMorphism::map_edge` in the free-category
/// interpretation: each operation (edge in the causal graph) maps
/// to a document mutation (morphism in the category of documents).
/// Apply a single operation to a document, producing the next state.
///
/// Returns the original document on error (e.g. node not yet
/// created during materialization of an out-of-order op).
fn apply_op(doc: &Document, op: &Op) -> Document {
    let result = match &op.action {
        Action::SetKey { node, key, value } => {
            doc.set_key(*node, key, value, op.tag.replica(), op.tag.timestamp())
        }
        Action::DeleteKey { node, key } => doc.delete_key(*node, key),
        Action::ListInsert {
            node,
            origin,
            value,
        } => doc.list_insert(*node, *origin, value.clone(), op.tag.replica(), op.tag.timestamp()),
        Action::ListDelete { node, element_tag } => doc.list_delete(*node, *element_tag),
        Action::CreateMap => doc
            .create_map(op.tag.replica(), op.tag.timestamp())
            .map(|(d, _)| d),
        Action::CreateList => doc
            .create_list(op.tag.replica(), op.tag.timestamp())
            .map(|(d, _)| d),
    };
    result.unwrap_or_else(|_| doc.clone())
}

// ---------------------------------------------------------------------------
// Semilattice
// ---------------------------------------------------------------------------

impl PartialEq for OpLog {
    fn eq(&self, other: &Self) -> bool {
        self.ops == other.ops && self.deps == other.deps
    }
}

impl Eq for OpLog {}

impl PartialOrd for OpLog {
    /// Subset order: `a <= b` iff every operation in `a` is also in
    /// `b` with the same dependencies.
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        use core::cmp::Ordering;
        let self_sub = self
            .ops
            .iter()
            .all(|(tag, op)| other.ops.get(tag) == Some(op))
            && self
                .deps
                .iter()
                .all(|(tag, d)| other.deps.get(tag) == Some(d));
        let other_sub = other
            .ops
            .iter()
            .all(|(tag, op)| self.ops.get(tag) == Some(op))
            && other
                .deps
                .iter()
                .all(|(tag, d)| self.deps.get(tag) == Some(d));
        match (self_sub, other_sub) {
            (true, true) => Some(Ordering::Equal),
            (true, false) => Some(Ordering::Less),
            (false, true) => Some(Ordering::Greater),
            (false, false) => None,
        }
    }
}

impl JoinSemilattice for OpLog {
    /// Union of operations and union of causal edges.
    ///
    /// Heads are recomputed from the merged graph.
    fn join(&self, other: &Self) -> Self {
        let from_self = self.ops.iter().map(|(k, v)| (*k, v.clone()));
        let only_in_other = other
            .ops
            .iter()
            .filter(|(k, _)| !self.ops.contains_key(k))
            .map(|(k, v)| (*k, v.clone()));
        let ops: BTreeMap<Tag, Op> = from_self.chain(only_in_other).collect();

        let deps_from_self = self.deps.iter().map(|(k, v)| (*k, v.clone()));
        let deps_only_in_other = other
            .deps
            .iter()
            .filter(|(k, _)| !self.deps.contains_key(k))
            .map(|(k, v)| (*k, v.clone()));
        let deps: BTreeMap<Tag, BTreeSet<Tag>> =
            deps_from_self.chain(deps_only_in_other).collect();

        let heads = Self::recompute_heads(&ops, &deps);
        Self { ops, deps, heads }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn r(id: u64) -> ReplicaId {
        ReplicaId::new(id)
    }

    fn t(ts: u64) -> Timestamp {
        Timestamp::new(ts)
    }

    #[test]
    fn empty_log_materializes_to_empty_document() -> Result<(), Error> {
        let log = OpLog::empty();
        let doc = log.materialize();
        assert!(doc.keys(NodeId::Root)?.is_empty());
        Ok(())
    }

    #[test]
    fn single_set_key_materializes() -> Result<(), Error> {
        let log = OpLog::empty().append(
            Action::SetKey {
                node: NodeId::Root,
                key: "name".into(),
                value: Value::Str("alice".into()),
            },
            r(0),
            t(1),
        )?;
        let doc = log.materialize();
        assert!(doc.get_key(NodeId::Root, "name")?.contains(&Value::Str("alice".into())));
        Ok(())
    }

    #[test]
    fn sequential_ops_applied_in_causal_order() -> Result<(), Error> {
        let log = OpLog::empty()
            .append(
                Action::SetKey {
                    node: NodeId::Root,
                    key: "x".into(),
                    value: Value::Int(1),
                },
                r(0),
                t(1),
            )?
            .append(
                Action::SetKey {
                    node: NodeId::Root,
                    key: "x".into(),
                    value: Value::Int(2),
                },
                r(0),
                t(2),
            )?;
        let doc = log.materialize();
        let vals = doc.get_key(NodeId::Root, "x")?;
        assert_eq!(vals.len(), 1);
        assert!(vals.contains(&Value::Int(2)));
        Ok(())
    }

    #[test]
    fn concurrent_ops_produce_mv_register_conflict() -> Result<(), Error> {
        let base = OpLog::empty().append(
            Action::SetKey {
                node: NodeId::Root,
                key: "x".into(),
                value: Value::Int(0),
            },
            r(0),
            t(1),
        )?;
        let base_heads = base.heads().clone();

        let left = base.append_with_deps(
            Action::SetKey {
                node: NodeId::Root,
                key: "x".into(),
                value: Value::Int(1),
            },
            r(0),
            t(2),
            base_heads.clone(),
        )?;
        let right = base.append_with_deps(
            Action::SetKey {
                node: NodeId::Root,
                key: "x".into(),
                value: Value::Int(2),
            },
            r(1),
            t(2),
            base_heads,
        )?;

        let merged = left.join(&right);
        let doc = merged.materialize();
        let vals = doc.get_key(NodeId::Root, "x")?;
        assert_eq!(vals.len(), 2);
        assert!(vals.contains(&Value::Int(1)));
        assert!(vals.contains(&Value::Int(2)));
        Ok(())
    }

    #[test]
    fn diff_returns_missing_ops() -> Result<(), Error> {
        let a = OpLog::empty().append(
            Action::SetKey {
                node: NodeId::Root,
                key: "x".into(),
                value: Value::Int(1),
            },
            r(0),
            t(1),
        )?;
        let b = OpLog::empty();
        let missing = a.diff(&b);
        assert_eq!(missing.len(), 1);
        assert_eq!(missing[0].tag(), Tag::new(r(0), t(1)));
        Ok(())
    }

    #[test]
    fn heads_track_frontier() -> Result<(), Error> {
        let log = OpLog::empty()
            .append(
                Action::SetKey {
                    node: NodeId::Root,
                    key: "x".into(),
                    value: Value::Int(1),
                },
                r(0),
                t(1),
            )?
            .append(
                Action::SetKey {
                    node: NodeId::Root,
                    key: "y".into(),
                    value: Value::Int(2),
                },
                r(0),
                t(2),
            )?;
        assert_eq!(log.heads().len(), 1);
        assert!(log.heads().contains(&Tag::new(r(0), t(2))));
        Ok(())
    }

    #[test]
    fn merged_logs_have_correct_heads() -> Result<(), Error> {
        let base_heads = BTreeSet::new();
        let a = OpLog::empty().append_with_deps(
            Action::SetKey {
                node: NodeId::Root,
                key: "x".into(),
                value: Value::Int(1),
            },
            r(0),
            t(1),
            base_heads.clone(),
        )?;
        let b = OpLog::empty().append_with_deps(
            Action::SetKey {
                node: NodeId::Root,
                key: "y".into(),
                value: Value::Int(2),
            },
            r(1),
            t(1),
            base_heads,
        )?;
        let merged = a.join(&b);
        assert_eq!(merged.heads().len(), 2);
        Ok(())
    }

    #[test]
    fn create_map_and_set_nested_key() -> Result<(), Error> {
        let map_tag = Tag::new(r(0), t(1));
        let map_node = NodeId::Created(map_tag);
        let log = OpLog::empty()
            .append(Action::CreateMap, r(0), t(1))?
            .append(
                Action::SetKey {
                    node: NodeId::Root,
                    key: "child".into(),
                    value: Value::Map(map_node),
                },
                r(0),
                t(2),
            )?
            .append(
                Action::SetKey {
                    node: map_node,
                    key: "name".into(),
                    value: Value::Str("inner".into()),
                },
                r(0),
                t(3),
            )?;
        let doc = log.materialize();
        assert!(doc
            .get_key(map_node, "name")?
            .contains(&Value::Str("inner".into())));
        Ok(())
    }

    #[test]
    fn list_insert_via_oplog() -> Result<(), Error> {
        let list_tag = Tag::new(r(0), t(1));
        let list_node = NodeId::Created(list_tag);
        let log = OpLog::empty()
            .append(Action::CreateList, r(0), t(1))?
            .append(
                Action::SetKey {
                    node: NodeId::Root,
                    key: "items".into(),
                    value: Value::List(list_node),
                },
                r(0),
                t(2),
            )?
            .append(
                Action::ListInsert {
                    node: list_node,
                    origin: Origin::Head,
                    value: Value::Str("a".into()),
                },
                r(0),
                t(3),
            )?
            .append(
                Action::ListInsert {
                    node: list_node,
                    origin: Origin::Head,
                    value: Value::Str("b".into()),
                },
                r(0),
                t(4),
            )?;
        let doc = log.materialize();
        let elems = doc.list_elements(list_node)?;
        assert_eq!(elems, vec![&Value::Str("b".into()), &Value::Str("a".into())]);
        Ok(())
    }

    #[test]
    fn duplicate_op_returns_error() {
        let log = OpLog::empty()
            .append(
                Action::SetKey {
                    node: NodeId::Root,
                    key: "x".into(),
                    value: Value::Int(1),
                },
                r(0),
                t(1),
            )
            .unwrap_or(OpLog::empty());
        let result = log.append(
            Action::SetKey {
                node: NodeId::Root,
                key: "y".into(),
                value: Value::Int(2),
            },
            r(0),
            t(1),
        );
        assert!(result.is_err());
    }

    fn arb_oplog() -> impl Strategy<Value = OpLog> {
        proptest::collection::vec((0u64..3, 0u64..20), 0..5).prop_map(|ops| {
            ops.iter().fold(OpLog::empty(), |log, &(rid, ts)| {
                let val = (rid.wrapping_mul(31).wrapping_add(ts)) % 10;
                let key_idx = (rid.wrapping_mul(7).wrapping_add(ts)) % 3;
                let key = match key_idx {
                    0 => "a",
                    1 => "b",
                    _ => "c",
                };
                log.append_with_deps(
                    Action::SetKey {
                        node: NodeId::Root,
                        key: key.into(),
                        value: Value::Uint(val),
                    },
                    ReplicaId::new(rid),
                    Timestamp::new(ts),
                    BTreeSet::new(),
                )
                .unwrap_or(log)
            })
        })
    }

    proptest! {
        #[test]
        fn semilattice_join_is_commutative(
            a in arb_oplog(),
            b in arb_oplog(),
        ) {
            prop_assert_eq!(a.join(&b), b.join(&a));
        }

        #[test]
        fn semilattice_join_is_associative(
            a in arb_oplog(),
            b in arb_oplog(),
            c in arb_oplog(),
        ) {
            prop_assert_eq!(a.join(&b).join(&c), a.join(&b.join(&c)));
        }

        #[test]
        fn semilattice_join_is_idempotent(a in arb_oplog()) {
            prop_assert_eq!(a.join(&a), a);
        }

        #[test]
        fn semilattice_join_is_upper_bound(
            a in arb_oplog(),
            b in arb_oplog(),
        ) {
            let joined = a.join(&b);
            prop_assert!(a <= joined);
            prop_assert!(b <= joined);
        }

        #[test]
        fn append_is_monotonic(
            a in arb_oplog(),
            rid in 0u64..3,
            ts in 0u64..100,
            key in "[a-c]{1}",
        ) {
            let val = (rid.wrapping_mul(31).wrapping_add(ts)) % 10;
            let b = a.append(
                Action::SetKey {
                    node: NodeId::Root,
                    key,
                    value: Value::Uint(val),
                },
                ReplicaId::new(rid),
                Timestamp::new(ts),
            )
            .unwrap_or(a.clone());
            prop_assert!(a <= b);
        }

        #[test]
        fn sequential_materialize_matches_direct(
            ops_raw in proptest::collection::vec((0u64..3, "[a-c]{1}"), 0..5),
        ) {
            let ops: Vec<(u64, u64, String)> = ops_raw
                .iter()
                .enumerate()
                .map(|(i, (rid, key))| (*rid, i as u64, key.clone()))
                .collect();
            let log = ops.iter().fold(OpLog::empty(), |log, (rid, ts, key)| {
                let val = (rid.wrapping_mul(31).wrapping_add(*ts)) % 10;
                log.append(
                    Action::SetKey {
                        node: NodeId::Root,
                        key: key.clone(),
                        value: Value::Uint(val),
                    },
                    ReplicaId::new(*rid),
                    Timestamp::new(*ts),
                )
                .unwrap_or(log)
            });
            let direct = ops.iter().fold(Document::new(), |doc, (rid, ts, key)| {
                let val = (rid.wrapping_mul(31).wrapping_add(*ts)) % 10;
                doc.set_key(
                    NodeId::Root,
                    key,
                    &Value::Uint(val),
                    ReplicaId::new(*rid),
                    Timestamp::new(*ts),
                )
                .unwrap_or(doc)
            });
            prop_assert_eq!(log.materialize(), direct);
        }
    }
}
