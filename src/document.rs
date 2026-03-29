//! Document: a CRDT tree of maps and lists.
//!
//! Composes [`MvRegister`] (for map key-value pairs) and [`Rga`]
//! (for ordered lists) into a nested document whose merge is a
//! componentwise join of all container nodes.
//!
//! The document is a flat map from [`NodeId`] to [`Node`], where
//! each node is either a map or a list.  Values reference nested
//! containers by [`NodeId`], forming a logical tree without
//! recursive types.
//!
//! The state space forms a join-semilattice as a product of the
//! individual node semilattices, keyed by [`NodeId`].

use std::collections::{BTreeMap, BTreeSet};

use comp_cat_rs::foundation::JoinSemilattice;
use serde::{Deserialize, Serialize};
use crate::error::Error;
use crate::mvregister::MvRegister;
use crate::rga::{Origin, Rga};
use crate::replica::{ReplicaId, Tag, Timestamp};

// ---------------------------------------------------------------------------
// Float64: total-ordered f64
// ---------------------------------------------------------------------------

/// A 64-bit float with total ordering (IEEE 754 `totalOrder`).
///
/// Wraps `f64` and implements [`Ord`] via [`f64::total_cmp`],
/// giving a deterministic comparison for all values including
/// NaN and signed zeros.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Float64(f64);

impl Float64 {
    /// Wrap an `f64`.
    #[must_use]
    pub fn new(v: f64) -> Self {
        Self(v)
    }

    /// The underlying `f64`.
    #[must_use]
    pub fn value(self) -> f64 {
        self.0
    }
}

impl PartialEq for Float64 {
    fn eq(&self, other: &Self) -> bool {
        self.0.total_cmp(&other.0) == core::cmp::Ordering::Equal
    }
}

impl Eq for Float64 {}

impl PartialOrd for Float64 {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Float64 {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

impl core::hash::Hash for Float64 {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state);
    }
}

// ---------------------------------------------------------------------------
// NodeId, Value, Node
// ---------------------------------------------------------------------------

/// Identity of a container node in the document.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum NodeId {
    /// The document root (always a map).
    Root,
    /// A container created by a specific operation.
    Created(Tag),
}

/// A scalar or container-reference value.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Value {
    /// JSON null.
    Null,
    /// Boolean.
    Bool(bool),
    /// Signed integer.
    Int(i64),
    /// Unsigned integer.
    Uint(u64),
    /// Floating-point (total-ordered).
    Float(Float64),
    /// String scalar.
    Str(String),
    /// Reference to a nested map container.
    Map(NodeId),
    /// Reference to a nested list container.
    List(NodeId),
    /// Reference to a nested text container.
    Text(NodeId),
}

/// A container node: either a map or a list.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Node {
    /// A map from string keys to multi-value registers of [`Value`].
    Map(BTreeMap<String, MvRegister<Value>>),
    /// An ordered list of [`Value`]s.
    List(Rga<Value>),
}

impl PartialEq for Node {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Map(a), Self::Map(b)) => a == b,
            (Self::List(a), Self::List(b)) => a == b,
            _ => false,
        }
    }
}

impl Eq for Node {}

impl PartialOrd for Node {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        match (self, other) {
            (Self::Map(a), Self::Map(b)) => map_partial_cmp(a, b),
            (Self::List(a), Self::List(b)) => a.partial_cmp(b),
            _ => None,
        }
    }
}

/// Subset order on maps of `MvRegister`s.
fn map_partial_cmp(
    a: &BTreeMap<String, MvRegister<Value>>,
    b: &BTreeMap<String, MvRegister<Value>>,
) -> Option<core::cmp::Ordering> {
    use core::cmp::Ordering;
    let a_sub_b = a
        .iter()
        .all(|(k, v)| b.get(k).is_some_and(|bv| v <= bv));
    let b_sub_a = b
        .iter()
        .all(|(k, v)| a.get(k).is_some_and(|av| v <= av));
    match (a_sub_b, b_sub_a) {
        (true, true) => Some(Ordering::Equal),
        (true, false) => Some(Ordering::Less),
        (false, true) => Some(Ordering::Greater),
        (false, false) => None,
    }
}

/// Join two map nodes by joining each key's register.
fn join_maps(
    a: &BTreeMap<String, MvRegister<Value>>,
    b: &BTreeMap<String, MvRegister<Value>>,
) -> BTreeMap<String, MvRegister<Value>> {
    let from_a = a.iter().map(|(k, av)| {
        let merged = b
            .get(k)
            .map_or_else(|| av.clone(), |bv| av.join(bv));
        (k.clone(), merged)
    });
    let only_in_b = b
        .iter()
        .filter(|(k, _)| !a.contains_key(*k))
        .map(|(k, v)| (k.clone(), v.clone()));
    from_a.chain(only_in_b).collect()
}

/// Join two nodes (must be the same variant).
fn join_nodes(a: &Node, b: &Node) -> Node {
    match (a, b) {
        (Node::Map(am), Node::Map(bm)) => Node::Map(join_maps(am, bm)),
        (Node::List(al), Node::List(bl)) => Node::List(al.join(bl)),
        _ => a.clone(),
    }
}

// ---------------------------------------------------------------------------
// Document
// ---------------------------------------------------------------------------

/// A CRDT document: a tree of maps and lists.
///
/// The root is always a map.  Nested containers are created via
/// [`create_map`](Self::create_map) / [`create_list`](Self::create_list)
/// and referenced by [`NodeId`] in [`Value::Map`] / [`Value::List`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    nodes: BTreeMap<NodeId, Node>,
    root: NodeId,
}

impl Default for Document {
    fn default() -> Self {
        Self::new()
    }
}

impl Document {
    /// A new document with an empty root map.
    #[must_use]
    pub fn new() -> Self {
        Self {
            nodes: core::iter::once((NodeId::Root, Node::Map(BTreeMap::new()))).collect(),
            root: NodeId::Root,
        }
    }

    /// The root node ID (always [`NodeId::Root`]).
    #[must_use]
    pub fn root(&self) -> NodeId {
        self.root
    }

    /// Compact all nodes, removing tombstoned entries in `safe_tags`.
    ///
    /// `safe_tags` should contain tags that all replicas have observed.
    #[must_use]
    pub fn compact(&self, safe_tags: &BTreeSet<Tag>) -> Self {
        Self {
            nodes: self
                .nodes
                .iter()
                .map(|(id, node)| {
                    let compacted = match node {
                        Node::Map(map) => Node::Map(
                            map.iter()
                                .map(|(k, reg)| (k.clone(), reg.compact(safe_tags)))
                                .collect(),
                        ),
                        Node::List(rga) => Node::List(rga.compact(safe_tags)),
                    };
                    (*id, compacted)
                })
                .collect(),
            root: self.root,
        }
    }

    // -- map operations -----------------------------------------------------

    /// Set a key in a map node, returning the new document.
    ///
    /// If the key already exists, the write follows
    /// [`MvRegister`] semantics (tombstones previous values).
    ///
    /// # Errors
    ///
    /// - [`Error::NodeNotFound`] if `node` does not exist.
    /// - [`Error::WrongNodeType`] if `node` is not a map.
    pub fn set_key(
        &self,
        node: NodeId,
        key: &str,
        value: &Value,
        replica: ReplicaId,
        timestamp: Timestamp,
    ) -> Result<Self, Error> {
        self.with_map_node(node, |map| {
            let register = map.get(key).map_or_else(
                || MvRegister::new(value.clone(), replica, timestamp),
                |reg| reg.write(value.clone(), replica, timestamp),
            );
            map.iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .filter(|(k, _)| k != key)
                .chain(core::iter::once((key.to_string(), register)))
                .collect()
        })
    }

    /// Delete a key from a map node, returning the new document.
    ///
    /// Tombstones all currently-observed values for the key.
    /// A concurrent write from another replica survives the merge.
    ///
    /// # Errors
    ///
    /// - [`Error::NodeNotFound`] if `node` does not exist.
    /// - [`Error::WrongNodeType`] if `node` is not a map.
    pub fn delete_key(&self, node: NodeId, key: &str) -> Result<Self, Error> {
        self.with_map_node(node, |map| {
            map.get(key).map_or_else(
                || map.clone(),
                |reg| {
                    let cleared = reg.clear();
                    map.iter()
                        .map(|(k, v)| (k.clone(), v.clone()))
                        .filter(|(k, _)| k != key)
                        .chain(core::iter::once((key.to_string(), cleared)))
                        .collect()
                },
            )
        })
    }

    /// Get the observed values for a key in a map node.
    ///
    /// Returns the [`MvRegister`]'s current values: one value if
    /// resolved, multiple if there are concurrent writes, empty
    /// if the key does not exist or was deleted.
    ///
    /// # Errors
    ///
    /// - [`Error::NodeNotFound`] if `node` does not exist.
    /// - [`Error::WrongNodeType`] if `node` is not a map.
    pub fn get_key(&self, node: NodeId, key: &str) -> Result<BTreeSet<&Value>, Error> {
        self.require_map(node).map(|map| {
            map.get(key)
                .map_or_else(BTreeSet::new, MvRegister::values)
        })
    }

    /// The keys with at least one observed value in a map node.
    ///
    /// # Errors
    ///
    /// - [`Error::NodeNotFound`] if `node` does not exist.
    /// - [`Error::WrongNodeType`] if `node` is not a map.
    pub fn keys(&self, node: NodeId) -> Result<BTreeSet<&str>, Error> {
        self.require_map(node).map(|map| {
            map.iter()
                .filter(|(_, reg)| !reg.is_empty())
                .map(|(k, _)| k.as_str())
                .collect()
        })
    }

    // -- list operations ----------------------------------------------------

    /// Insert a value into a list node, returning the new document.
    ///
    /// # Errors
    ///
    /// - [`Error::NodeNotFound`] if `node` does not exist.
    /// - [`Error::WrongNodeType`] if `node` is not a list.
    pub fn list_insert(
        &self,
        node: NodeId,
        origin: Origin,
        value: Value,
        replica: ReplicaId,
        timestamp: Timestamp,
    ) -> Result<Self, Error> {
        self.with_list_node(node, |list| {
            list.insert_after(origin, value, replica, timestamp)
        })
    }

    /// Delete an element from a list node by tag, returning the new document.
    ///
    /// # Errors
    ///
    /// - [`Error::NodeNotFound`] if `node` does not exist.
    /// - [`Error::WrongNodeType`] if `node` is not a list.
    pub fn list_delete(&self, node: NodeId, tag: Tag) -> Result<Self, Error> {
        self.with_list_node(node, |list| list.delete(tag))
    }

    /// The ordered elements of a list node.
    ///
    /// # Errors
    ///
    /// - [`Error::NodeNotFound`] if `node` does not exist.
    /// - [`Error::WrongNodeType`] if `node` is not a list.
    pub fn list_elements(&self, node: NodeId) -> Result<Vec<&Value>, Error> {
        self.require_list(node).map(Rga::elements)
    }

    // -- container creation -------------------------------------------------

    /// Create a new empty map container, returning the updated
    /// document and the new node's ID.
    ///
    /// Use the returned [`NodeId`] in [`Value::Map`] to reference
    /// the container from a parent map or list.
    ///
    /// # Errors
    ///
    /// - [`Error::NodeAlreadyExists`] if the tag collides with
    ///   an existing node.
    pub fn create_map(
        &self,
        replica: ReplicaId,
        timestamp: Timestamp,
    ) -> Result<(Self, NodeId), Error> {
        let id = NodeId::Created(Tag::new(replica, timestamp));
        if self.nodes.contains_key(&id) {
            Err(Error::NodeAlreadyExists { node: id })
        } else {
            Ok((
                Self {
                    nodes: self
                        .nodes
                        .iter()
                        .map(|(k, v)| (*k, v.clone()))
                        .chain(core::iter::once((id, Node::Map(BTreeMap::new()))))
                        .collect(),
                    root: self.root,
                },
                id,
            ))
        }
    }

    /// Create a new empty list container, returning the updated
    /// document and the new node's ID.
    ///
    /// # Errors
    ///
    /// - [`Error::NodeAlreadyExists`] if the tag collides with
    ///   an existing node.
    pub fn create_list(
        &self,
        replica: ReplicaId,
        timestamp: Timestamp,
    ) -> Result<(Self, NodeId), Error> {
        let id = NodeId::Created(Tag::new(replica, timestamp));
        if self.nodes.contains_key(&id) {
            Err(Error::NodeAlreadyExists { node: id })
        } else {
            Ok((
                Self {
                    nodes: self
                        .nodes
                        .iter()
                        .map(|(k, v)| (*k, v.clone()))
                        .chain(core::iter::once((id, Node::List(Rga::empty()))))
                        .collect(),
                    root: self.root,
                },
                id,
            ))
        }
    }

    // -- internal -----------------------------------------------------------

    /// Get a reference to a map node, or error.
    fn require_map(&self, node: NodeId) -> Result<&BTreeMap<String, MvRegister<Value>>, Error> {
        self.nodes
            .get(&node)
            .ok_or(Error::NodeNotFound { node })
            .and_then(|n| match n {
                Node::Map(map) => Ok(map),
                Node::List(_) => Err(Error::WrongNodeType {
                    node,
                    expected: "map",
                }),
            })
    }

    /// Get a reference to a list node, or error.
    fn require_list(&self, node: NodeId) -> Result<&Rga<Value>, Error> {
        self.nodes
            .get(&node)
            .ok_or(Error::NodeNotFound { node })
            .and_then(|n| match n {
                Node::List(list) => Ok(list),
                Node::Map(_) => Err(Error::WrongNodeType {
                    node,
                    expected: "list",
                }),
            })
    }

    /// Apply a function to a map node, replacing it in the document.
    fn with_map_node(
        &self,
        target: NodeId,
        f: impl FnOnce(&BTreeMap<String, MvRegister<Value>>) -> BTreeMap<String, MvRegister<Value>>,
    ) -> Result<Self, Error> {
        self.require_map(target).map(|map| {
            let updated = Node::Map(f(map));
            Self {
                nodes: self
                    .nodes
                    .iter()
                    .map(|(id, n)| {
                        if *id == target { (*id, updated.clone()) } else { (*id, n.clone()) }
                    })
                    .collect(),
                root: self.root,
            }
        })
    }

    /// Apply a function to a list node, replacing it in the document.
    fn with_list_node(
        &self,
        target: NodeId,
        f: impl FnOnce(&Rga<Value>) -> Rga<Value>,
    ) -> Result<Self, Error> {
        self.require_list(target).map(|list| {
            let updated = Node::List(f(list));
            Self {
                nodes: self
                    .nodes
                    .iter()
                    .map(|(id, n)| {
                        if *id == target { (*id, updated.clone()) } else { (*id, n.clone()) }
                    })
                    .collect(),
                root: self.root,
            }
        })
    }
}

// ---------------------------------------------------------------------------
// Semilattice impls
// ---------------------------------------------------------------------------

impl PartialEq for Document {
    fn eq(&self, other: &Self) -> bool {
        self.root == other.root && self.nodes == other.nodes
    }
}

impl Eq for Document {}

impl PartialOrd for Document {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        use core::cmp::Ordering;
        (self.root == other.root)
            .then(|| {
                let self_sub = self
                    .nodes
                    .iter()
                    .all(|(id, n)| other.nodes.get(id).is_some_and(|on| n <= on));
                let other_sub = other
                    .nodes
                    .iter()
                    .all(|(id, n)| self.nodes.get(id).is_some_and(|sn| n <= sn));
                match (self_sub, other_sub) {
                    (true, true) => Some(Ordering::Equal),
                    (true, false) => Some(Ordering::Less),
                    (false, true) => Some(Ordering::Greater),
                    (false, false) => None,
                }
            })
            .flatten()
    }
}

impl JoinSemilattice for Document {
    /// Componentwise join of all container nodes.
    fn join(&self, other: &Self) -> Self {
        let from_self = self.nodes.iter().map(|(id, sn)| {
            let merged = other
                .nodes
                .get(id)
                .map_or_else(|| sn.clone(), |on| join_nodes(sn, on));
            (*id, merged)
        });
        let only_in_other = other
            .nodes
            .iter()
            .filter(|(id, _)| !self.nodes.contains_key(id))
            .map(|(id, n)| (*id, n.clone()));
        Self {
            nodes: from_self.chain(only_in_other).collect(),
            root: self.root,
        }
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

    // -- deterministic tests ------------------------------------------------

    #[test]
    fn new_document_has_empty_root_map() -> Result<(), Error> {
        let doc = Document::new();
        assert!(doc.keys(NodeId::Root)?.is_empty());
        Ok(())
    }

    #[test]
    fn set_and_get_key() -> Result<(), Error> {
        let doc = Document::new()
            .set_key(NodeId::Root, "name", &Value::Str("alice".into()), r(0), t(1))?;
        let vals = doc.get_key(NodeId::Root, "name")?;
        assert_eq!(vals.len(), 1);
        assert!(vals.contains(&Value::Str("alice".into())));
        Ok(())
    }

    #[test]
    fn sequential_writes_replace_value() -> Result<(), Error> {
        let doc = Document::new()
            .set_key(NodeId::Root, "x", &Value::Int(1), r(0), t(1))?
            .set_key(NodeId::Root, "x", &Value::Int(2), r(0), t(2))?;
        let vals = doc.get_key(NodeId::Root, "x")?;
        assert_eq!(vals.len(), 1);
        assert!(vals.contains(&Value::Int(2)));
        Ok(())
    }

    #[test]
    fn concurrent_writes_produce_multiple_values() -> Result<(), Error> {
        let base = Document::new()
            .set_key(NodeId::Root, "x", &Value::Int(0), r(0), t(1))?;
        let a = base.set_key(NodeId::Root, "x", &Value::Int(1), r(0), t(2))?;
        let b = base.set_key(NodeId::Root, "x", &Value::Int(2), r(1), t(2))?;
        let merged = a.join(&b);
        let vals = merged.get_key(NodeId::Root, "x")?;
        assert_eq!(vals.len(), 2);
        assert!(vals.contains(&Value::Int(1)));
        assert!(vals.contains(&Value::Int(2)));
        Ok(())
    }

    #[test]
    fn delete_key_removes_value() -> Result<(), Error> {
        let doc = Document::new()
            .set_key(NodeId::Root, "x", &Value::Int(1), r(0), t(1))?
            .delete_key(NodeId::Root, "x")?;
        assert!(doc.get_key(NodeId::Root, "x")?.is_empty());
        assert!(!doc.keys(NodeId::Root)?.contains("x"));
        Ok(())
    }

    #[test]
    fn concurrent_set_survives_remote_delete() -> Result<(), Error> {
        let base = Document::new()
            .set_key(NodeId::Root, "x", &Value::Int(0), r(0), t(1))?;
        let deleted = base.delete_key(NodeId::Root, "x")?;
        let written = base.set_key(NodeId::Root, "x", &Value::Int(42), r(1), t(2))?;
        let merged = deleted.join(&written);
        let vals = merged.get_key(NodeId::Root, "x")?;
        assert!(vals.contains(&Value::Int(42)));
        Ok(())
    }

    #[test]
    fn nested_map_operations() -> Result<(), Error> {
        let doc = Document::new();
        let (doc, child_id) = doc.create_map(r(0), t(1))?;
        let doc = doc
            .set_key(NodeId::Root, "child", &Value::Map(child_id), r(0), t(2))?
            .set_key(child_id, "name", &Value::Str("inner".into()), r(0), t(3))?;
        let child_vals = doc.get_key(child_id, "name")?;
        assert!(child_vals.contains(&Value::Str("inner".into())));
        Ok(())
    }

    #[test]
    fn list_insert_and_elements() -> Result<(), Error> {
        let doc = Document::new();
        let (doc, list_id) = doc.create_list(r(0), t(1))?;
        let doc = doc
            .set_key(NodeId::Root, "items", &Value::List(list_id), r(0), t(2))?
            .list_insert(list_id, Origin::Head, Value::Str("a".into()), r(0), t(3))?
            .list_insert(list_id, Origin::Head, Value::Str("b".into()), r(0), t(4))?;
        let elems = doc.list_elements(list_id)?;
        assert_eq!(elems, vec![&Value::Str("b".into()), &Value::Str("a".into())]);
        Ok(())
    }

    #[test]
    fn merge_documents_with_disjoint_keys() -> Result<(), Error> {
        let base = Document::new();
        let a = base.set_key(NodeId::Root, "x", &Value::Int(1), r(0), t(1))?;
        let b = base.set_key(NodeId::Root, "y", &Value::Int(2), r(1), t(1))?;
        let merged = a.join(&b);
        assert!(merged.get_key(NodeId::Root, "x")?.contains(&Value::Int(1)));
        assert!(merged.get_key(NodeId::Root, "y")?.contains(&Value::Int(2)));
        Ok(())
    }

    #[test]
    fn merge_documents_with_nested_containers() -> Result<(), Error> {
        let doc = Document::new();
        let (doc, map_a) = doc.create_map(r(0), t(1))?;
        let (doc, map_b) = doc.create_map(r(1), t(1))?;

        let left = doc
            .clone()
            .set_key(NodeId::Root, "a", &Value::Map(map_a), r(0), t(2))?
            .set_key(map_a, "val", &Value::Int(10), r(0), t(3))?;

        let right = doc
            .set_key(NodeId::Root, "b", &Value::Map(map_b), r(1), t(2))?
            .set_key(map_b, "val", &Value::Int(20), r(1), t(3))?;

        let merged = left.join(&right);
        assert!(merged.get_key(map_a, "val")?.contains(&Value::Int(10)));
        assert!(merged.get_key(map_b, "val")?.contains(&Value::Int(20)));
        Ok(())
    }

    #[test]
    fn wrong_node_type_returns_error() {
        let doc = Document::new();
        let result = doc.list_insert(
            NodeId::Root,
            Origin::Head,
            Value::Int(1),
            r(0),
            t(1),
        );
        assert!(result.is_err());
    }

    #[test]
    fn missing_node_returns_error() {
        let doc = Document::new();
        let fake = NodeId::Created(Tag::new(r(99), t(99)));
        assert!(doc.get_key(fake, "x").is_err());
    }

    // -- property-based tests -----------------------------------------------

    fn arb_document() -> impl Strategy<Value = Document> {
        proptest::collection::vec((0u64..3, 0u64..20, "[a-c]{1}"), 0..5).prop_map(
            |ops| {
                ops.iter().fold(Document::new(), |doc, (rid, ts, key)| {
                    let val = (rid.wrapping_mul(31).wrapping_add(*ts)) % 10;
                    // set_key on Root always succeeds
                    doc.set_key(
                        NodeId::Root,
                        key,
                        &Value::Uint(val),
                        ReplicaId::new(*rid),
                        Timestamp::new(*ts),
                    )
                    .unwrap_or(doc)
                })
            },
        )
    }

    proptest! {
        #[test]
        fn semilattice_join_is_commutative(
            a in arb_document(),
            b in arb_document(),
        ) {
            prop_assert_eq!(a.join(&b), b.join(&a));
        }

        #[test]
        fn semilattice_join_is_associative(
            a in arb_document(),
            b in arb_document(),
            c in arb_document(),
        ) {
            prop_assert_eq!(a.join(&b).join(&c), a.join(&b.join(&c)));
        }

        #[test]
        fn semilattice_join_is_idempotent(a in arb_document()) {
            prop_assert_eq!(a.join(&a), a);
        }

        #[test]
        fn semilattice_join_is_upper_bound(
            a in arb_document(),
            b in arb_document(),
        ) {
            let joined = a.join(&b);
            prop_assert!(a <= joined);
            prop_assert!(b <= joined);
        }

        #[test]
        fn set_key_is_monotonic(
            a in arb_document(),
            rid in 0u64..3,
            ts in 0u64..100,
            key in "[a-c]{1}",
        ) {
            let val = (rid.wrapping_mul(31).wrapping_add(ts)) % 10;
            let b = a.set_key(
                NodeId::Root,
                &key,
                &Value::Uint(val),
                ReplicaId::new(rid),
                Timestamp::new(ts),
            )
            .unwrap_or(a.clone());
            prop_assert!(a <= b);
        }
    }
}
