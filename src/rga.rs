//! Replicated Growable Array: a sequence CRDT.
//!
//! Each element is tagged with a unique [`Tag`] and records the
//! [`Origin`] it was inserted after.  Elements form a tree rooted
//! at the virtual head; the visible sequence is a DFS traversal
//! where siblings are visited in descending tag order (later
//! concurrent inserts appear first).
//!
//! The state space forms a join-semilattice as a product of two
//! grow-only structures (entries + tombstones), identical to
//! [`OrSet`](crate::orset::OrSet) and
//! [`MvRegister`](crate::mvregister::MvRegister).
//!
//! Deletion tombstones the tag but preserves the tree structure:
//! children of a deleted element remain visible, appearing in the
//! position their parent formerly occupied.

use std::collections::{BTreeMap, BTreeSet};

use comp_cat_rs::foundation::JoinSemilattice;
use serde::{Deserialize, Serialize};
use crate::replica::{ReplicaId, Tag, Timestamp};

/// The position after which an element was inserted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Origin {
    /// Inserted at the beginning (before all existing elements).
    Head,
    /// Inserted immediately after the element with this tag.
    After(Tag),
}

impl core::fmt::Display for Origin {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Head => write!(f, "head"),
            Self::After(tag) => write!(f, "after:{tag}"),
        }
    }
}

/// An internal entry in the RGA.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct RgaEntry<A> {
    value: A,
    origin: Origin,
}

/// A replicated growable array (`Rga`).
///
/// Maintains insertion order across distributed replicas.
/// Concurrent inserts at the same position are ordered
/// deterministically by tag (descending).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Rga<A> {
    elements: BTreeMap<Tag, RgaEntry<A>>,
    tombstones: BTreeSet<Tag>,
}

impl<A> Rga<A> {
    /// An empty sequence.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            elements: BTreeMap::new(),
            tombstones: BTreeSet::new(),
        }
    }

    /// Insert a value after the given origin, returning the new sequence.
    ///
    /// If the tag already exists, the insert is a no-op (tags must
    /// be globally unique).
    #[must_use]
    pub fn insert_after(
        &self,
        origin: Origin,
        value: A,
        replica: ReplicaId,
        timestamp: Timestamp,
    ) -> Self
    where
        A: Clone,
    {
        let tag = Tag::new(replica, timestamp);
        if self.elements.contains_key(&tag) {
            self.clone()
        } else {
            Self {
                elements: self
                    .elements
                    .iter()
                    .map(|(k, v)| (*k, v.clone()))
                    .chain(core::iter::once((tag, RgaEntry { value, origin })))
                    .collect(),
                tombstones: self.tombstones.clone(),
            }
        }
    }

    /// Delete the element with the given tag, returning the new sequence.
    ///
    /// Children of the deleted element remain visible in the
    /// sequence.  If the tag is already tombstoned or does not
    /// exist, the delete is a no-op.
    #[must_use]
    pub fn delete(&self, tag: Tag) -> Self
    where
        A: Clone,
    {
        let already_dead =
            self.tombstones.contains(&tag) || !self.elements.contains_key(&tag);
        if already_dead {
            self.clone()
        } else {
            Self {
                elements: self.elements.clone(),
                tombstones: self
                    .tombstones
                    .iter()
                    .copied()
                    .chain(core::iter::once(tag))
                    .collect(),
            }
        }
    }

    /// Remove tombstoned entries that all replicas have observed.
    ///
    /// When a tombstoned entry is removed, its children are
    /// re-parented to the removed entry's origin so the tree
    /// structure remains valid.
    #[must_use]
    pub fn compact(&self, safe_tags: &BTreeSet<Tag>) -> Self
    where
        A: Clone,
    {
        let removable: BTreeSet<Tag> = self
            .tombstones
            .intersection(safe_tags)
            .filter(|tag| self.elements.contains_key(tag))
            .copied()
            .collect();

        // Build a map from removed tag -> its origin, for re-parenting.
        let reparent: BTreeMap<Tag, Origin> = removable
            .iter()
            .filter_map(|tag| {
                self.elements
                    .get(tag)
                    .map(|entry| (*tag, entry.origin))
            })
            .collect();

        // For remaining entries, re-parent any whose origin points
        // to a removed entry.
        let entries: BTreeMap<Tag, RgaEntry<A>> = self
            .elements
            .iter()
            .filter(|(tag, _)| !removable.contains(tag))
            .map(|(tag, entry)| {
                let new_origin = Self::resolve_origin(entry.origin, &reparent);
                (*tag, RgaEntry {
                    value: entry.value.clone(),
                    origin: new_origin,
                })
            })
            .collect();

        Self {
            elements: entries,
            tombstones: self
                .tombstones
                .difference(&removable)
                .copied()
                .collect(),
        }
    }

    /// Chase re-parent links until we find an origin that is
    /// not being removed.
    fn resolve_origin(
        origin: Origin,
        reparent: &BTreeMap<Tag, Origin>,
    ) -> Origin {
        match origin {
            Origin::Head => Origin::Head,
            Origin::After(tag) => reparent
                .get(&tag)
                .map_or(origin, |parent_origin| {
                    Self::resolve_origin(*parent_origin, reparent)
                }),
        }
    }

    /// The ordered sequence of non-tombstoned values.
    #[must_use]
    pub fn elements(&self) -> Vec<&A> {
        self.ordered_tags(Origin::Head)
            .into_iter()
            .filter_map(|tag| self.elements.get(&tag).map(|e| &e.value))
            .collect()
    }

    /// The ordered sequence of `(tag, value)` pairs.
    ///
    /// Useful for obtaining the [`Tag`] at a given position, which
    /// is needed as an [`Origin`] for subsequent inserts.
    #[must_use]
    pub fn entries(&self) -> Vec<(Tag, &A)> {
        self.ordered_tags(Origin::Head)
            .into_iter()
            .filter_map(|tag| self.elements.get(&tag).map(|e| (tag, &e.value)))
            .collect()
    }

    /// The number of non-tombstoned elements.
    #[must_use]
    pub fn len(&self) -> usize {
        self.elements
            .keys()
            .filter(|tag| !self.tombstones.contains(tag))
            .count()
    }

    /// Whether the sequence has no visible elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.elements
            .keys()
            .all(|tag| self.tombstones.contains(tag))
    }

    /// DFS traversal of the insertion tree rooted at `parent`.
    ///
    /// Visits ALL children (including tombstoned) to preserve tree
    /// structure, but only emits non-tombstoned tags.  Siblings
    /// are visited in descending tag order (later inserts first).
    fn ordered_tags(&self, parent: Origin) -> Vec<Tag> {
        self.elements
            .iter()
            .filter(|(_, entry)| entry.origin == parent)
            .map(|(tag, _)| *tag)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .flat_map(|tag| {
                let subtree = self.ordered_tags(Origin::After(tag));
                if self.tombstones.contains(&tag) {
                    subtree
                } else {
                    core::iter::once(tag).chain(subtree).collect()
                }
            })
            .collect()
    }
}

impl<A: PartialEq> PartialEq for Rga<A> {
    fn eq(&self, other: &Self) -> bool {
        self.elements == other.elements && self.tombstones == other.tombstones
    }
}

impl<A: Eq> Eq for Rga<A> {}

impl<A: PartialEq> PartialOrd for Rga<A> {
    /// Product of subset orders on entries and tombstones.
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        use core::cmp::Ordering;
        let self_sub_other = self
            .elements
            .iter()
            .all(|(k, v)| other.elements.get(k) == Some(v))
            && self.tombstones.is_subset(&other.tombstones);
        let other_sub_self = other
            .elements
            .iter()
            .all(|(k, v)| self.elements.get(k) == Some(v))
            && other.tombstones.is_subset(&self.tombstones);
        match (self_sub_other, other_sub_self) {
            (true, true) => Some(Ordering::Equal),
            (true, false) => Some(Ordering::Less),
            (false, true) => Some(Ordering::Greater),
            (false, false) => None,
        }
    }
}

impl<A: Eq + Clone> JoinSemilattice for Rga<A> {
    /// Union of entries and union of tombstones.
    ///
    /// For shared tags, values and origins must be equal (invariant:
    /// a tag uniquely identifies one insertion).
    fn join(&self, other: &Self) -> Self {
        let from_self = self.elements.iter().map(|(k, v)| (*k, v.clone()));
        let only_in_other = other
            .elements
            .iter()
            .filter(|(k, _)| !self.elements.contains_key(k))
            .map(|(k, v)| (*k, v.clone()));
        Self {
            elements: from_self.chain(only_in_other).collect(),
            tombstones: self
                .tombstones
                .union(&other.tombstones)
                .copied()
                .collect(),
        }
    }
}

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

    fn element_for_tag(rid: u64, ts: u64) -> u64 {
        (rid.wrapping_mul(31).wrapping_add(ts)) % 5
    }

    #[test]
    fn empty_rga_has_no_elements() {
        let rga: Rga<u64> = Rga::empty();
        assert!(rga.is_empty());
        assert_eq!(rga.len(), 0);
        assert!(rga.elements().is_empty());
    }

    #[test]
    fn insert_at_head_creates_single_element() {
        let rga = Rga::empty().insert_after(Origin::Head, 42_u64, r(0), t(1));
        assert_eq!(rga.len(), 1);
        assert_eq!(rga.elements(), vec![&42]);
    }

    #[test]
    fn multiple_inserts_at_head_ordered_by_tag_descending() {
        // Insert A then B at head; B has higher tag, so B comes first
        let rga = Rga::empty()
            .insert_after(Origin::Head, 1_u64, r(0), t(1))
            .insert_after(Origin::Head, 2, r(0), t(2));
        assert_eq!(rga.elements(), vec![&2, &1]);
    }

    #[test]
    fn insert_after_existing_element() {
        let rga = Rga::empty()
            .insert_after(Origin::Head, 1_u64, r(0), t(1));
        let a_tag = Tag::new(r(0), t(1));
        let rga = rga.insert_after(Origin::After(a_tag), 2, r(0), t(2));
        // Sequence: 1, 2 (2 inserted after 1)
        assert_eq!(rga.elements(), vec![&1, &2]);
    }

    #[test]
    fn concurrent_inserts_at_same_position_ordered_by_tag() {
        let a_tag = Tag::new(r(0), t(1));
        let base = Rga::empty()
            .insert_after(Origin::Head, 1_u64, r(0), t(1));

        // Two replicas insert after A concurrently
        let left = base.insert_after(Origin::After(a_tag), 2, r(0), t(2));
        let right = base.insert_after(Origin::After(a_tag), 3, r(1), t(3));

        let merged = left.join(&right);
        // Children of A: tag(1,3)=3 and tag(0,2)=2
        // tag(1,3) > tag(0,2), so 3 comes first
        assert_eq!(merged.elements(), vec![&1, &3, &2]);
    }

    #[test]
    fn delete_removes_element_from_sequence() {
        let rga = Rga::empty()
            .insert_after(Origin::Head, 1_u64, r(0), t(1))
            .insert_after(Origin::Head, 2, r(0), t(2));
        let tag_a = Tag::new(r(0), t(1));
        let rga = rga.delete(tag_a);
        assert_eq!(rga.elements(), vec![&2]);
        assert_eq!(rga.len(), 1);
    }

    #[test]
    fn children_of_deleted_element_survive() {
        let a_tag = Tag::new(r(0), t(1));
        let rga = Rga::empty()
            .insert_after(Origin::Head, 1_u64, r(0), t(1))
            .insert_after(Origin::After(a_tag), 2, r(0), t(2))
            .delete(a_tag);
        // A is tombstoned, but B (child of A) survives
        assert_eq!(rga.elements(), vec![&2]);
    }

    #[test]
    fn concurrent_insert_survives_remote_delete() {
        let a_tag = Tag::new(r(0), t(1));
        let base = Rga::empty()
            .insert_after(Origin::Head, 1_u64, r(0), t(1));

        let deleted = base.delete(a_tag);
        let inserted = base.insert_after(Origin::After(a_tag), 2, r(1), t(2));

        let merged = deleted.join(&inserted);
        // A is tombstoned, B (child of A) survives
        assert_eq!(merged.elements(), vec![&2]);
    }

    #[test]
    fn join_merges_disjoint_sequences() {
        let a = Rga::empty().insert_after(Origin::Head, 1_u64, r(0), t(1));
        let b = Rga::empty().insert_after(Origin::Head, 2_u64, r(1), t(2));
        let merged = a.join(&b);
        assert_eq!(merged.len(), 2);
        // tag(1,2) > tag(0,1), so 2 comes first
        assert_eq!(merged.elements(), vec![&2, &1]);
    }

    #[test]
    fn entries_returns_tags_with_values() {
        let rga = Rga::empty()
            .insert_after(Origin::Head, 1_u64, r(0), t(1))
            .insert_after(Origin::Head, 2, r(0), t(2));
        let entries = rga.entries();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0], (Tag::new(r(0), t(2)), &2));
        assert_eq!(entries[1], (Tag::new(r(0), t(1)), &1));
    }

    fn arb_rga() -> impl Strategy<Value = Rga<u64>> {
        proptest::collection::vec((0u64..3, 0u64..20), 0..6).prop_map(|ops| {
            ops.iter().fold(Rga::empty(), |rga, &(rid, ts)| {
                rga.insert_after(
                    Origin::Head,
                    element_for_tag(rid, ts),
                    ReplicaId::new(rid),
                    Timestamp::new(ts),
                )
            })
        })
    }

    proptest! {
        #[test]
        fn semilattice_join_is_commutative(
            a in arb_rga(),
            b in arb_rga(),
        ) {
            prop_assert_eq!(a.join(&b), b.join(&a));
        }

        #[test]
        fn semilattice_join_is_associative(
            a in arb_rga(),
            b in arb_rga(),
            c in arb_rga(),
        ) {
            prop_assert_eq!(a.join(&b).join(&c), a.join(&b.join(&c)));
        }

        #[test]
        fn semilattice_join_is_idempotent(a in arb_rga()) {
            prop_assert_eq!(a.join(&a), a);
        }

        #[test]
        fn semilattice_join_is_upper_bound(
            a in arb_rga(),
            b in arb_rga(),
        ) {
            let joined = a.join(&b);
            prop_assert!(a <= joined);
            prop_assert!(b <= joined);
        }

        #[test]
        fn insert_is_monotonic(
            a in arb_rga(),
            rid in 0u64..3,
            ts in 0u64..100,
        ) {
            let b = a.insert_after(
                Origin::Head,
                element_for_tag(rid, ts),
                ReplicaId::new(rid),
                Timestamp::new(ts),
            );
            prop_assert!(a <= b);
        }

        #[test]
        fn delete_is_monotonic(a in arb_rga(), rid in 0u64..3, ts in 0u64..20) {
            let tag = Tag::new(ReplicaId::new(rid), Timestamp::new(ts));
            let b = a.delete(tag);
            prop_assert!(a <= b);
        }
    }
}
