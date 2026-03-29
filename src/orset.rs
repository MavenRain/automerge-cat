//! Observed-remove set: an add-wins set CRDT.
//!
//! Each addition is tagged with a unique [`Tag`] (replica + timestamp).
//! Removal tombstones only the tags known at removal time, so a
//! concurrent add with an unknown tag survives the merge (add wins).
//!
//! The state space forms a join-semilattice as a product of two
//! grow-only structures:
//! - `entries`: a grow-only map from [`Tag`] to element
//! - `tombstones`: a grow-only set of [`Tag`]s
//!
//! The partial order is componentwise subset inclusion.  The join
//! is componentwise union.  By `comp_cat_rs::collapse::join_is_colimit`,
//! this join is a left Kan extension.

use std::collections::{BTreeMap, BTreeSet};

use comp_cat_rs::foundation::JoinSemilattice;
use serde::{Deserialize, Serialize};
use crate::replica::{ReplicaId, Tag, Timestamp};

/// An observed-remove set (`ORSet`).
///
/// Elements of type `A` are tracked by unique [`Tag`]s.  An element
/// is "observed" (present) iff at least one of its tags is not
/// tombstoned.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrSet<A: Ord> {
    entries: BTreeMap<Tag, A>,
    tombstones: BTreeSet<Tag>,
}

impl<A: Ord> OrSet<A> {
    /// An empty set.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            entries: BTreeMap::new(),
            tombstones: BTreeSet::new(),
        }
    }

    /// Add an element, returning the new set.
    ///
    /// The caller supplies a unique `(replica, timestamp)` pair to
    /// tag this addition.  The tag must be globally unique; reusing
    /// a tag is a logic error.
    #[must_use]
    pub fn add(&self, element: A, replica: ReplicaId, timestamp: Timestamp) -> Self
    where
        A: Clone,
    {
        let tag = Tag::new(replica, timestamp);
        if self.entries.contains_key(&tag) {
            self.clone()
        } else {
            Self {
                entries: self
                    .entries
                    .iter()
                    .map(|(k, v)| (*k, v.clone()))
                    .chain(core::iter::once((tag, element)))
                    .collect(),
                tombstones: self.tombstones.clone(),
            }
        }
    }

    /// Remove all currently-observed occurrences of an element.
    ///
    /// Only tombstones tags visible in *this* replica's state.
    /// A concurrent add on another replica with an unknown tag
    /// will survive the merge (add-wins semantics).
    #[must_use]
    pub fn remove(&self, element: &A) -> Self
    where
        A: Clone,
    {
        let new_tombs: BTreeSet<Tag> = self
            .entries
            .iter()
            .filter(|(tag, v)| *v == element && !self.tombstones.contains(tag))
            .map(|(tag, _)| *tag)
            .collect();
        Self {
            entries: self.entries.clone(),
            tombstones: self
                .tombstones
                .union(&new_tombs)
                .copied()
                .collect(),
        }
    }

    /// Remove tombstoned entries that all replicas have observed.
    #[must_use]
    pub fn compact(&self, safe_tags: &BTreeSet<Tag>) -> Self
    where
        A: Clone,
    {
        let removable: BTreeSet<Tag> = self
            .tombstones
            .intersection(safe_tags)
            .copied()
            .collect();
        Self {
            entries: self
                .entries
                .iter()
                .filter(|(tag, _)| !removable.contains(tag))
                .map(|(k, v)| (*k, v.clone()))
                .collect(),
            tombstones: self
                .tombstones
                .difference(&removable)
                .copied()
                .collect(),
        }
    }

    /// The set of currently observed elements.
    #[must_use]
    pub fn elements(&self) -> BTreeSet<&A> {
        self.entries
            .iter()
            .filter(|(tag, _)| !self.tombstones.contains(tag))
            .map(|(_, v)| v)
            .collect()
    }

    /// Whether an element is currently observed.
    #[must_use]
    pub fn contains(&self, element: &A) -> bool {
        self.entries
            .iter()
            .any(|(tag, v)| v == element && !self.tombstones.contains(tag))
    }

    /// The number of distinct observed elements.
    #[must_use]
    pub fn len(&self) -> usize {
        self.elements().len()
    }

    /// Whether the set has no observed elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries
            .keys()
            .all(|tag| self.tombstones.contains(tag))
    }
}

impl<A: Ord> PartialEq for OrSet<A> {
    fn eq(&self, other: &Self) -> bool {
        self.entries == other.entries && self.tombstones == other.tombstones
    }
}

impl<A: Ord> Eq for OrSet<A> {}

impl<A: Ord> PartialOrd for OrSet<A> {
    /// Product of subset orders on entries and tombstones.
    ///
    /// `a <= b` iff every entry in `a` appears in `b` with the same
    /// value, AND every tombstone in `a` appears in `b`.
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        use core::cmp::Ordering;
        let self_sub_other = self
            .entries
            .iter()
            .all(|(k, v)| other.entries.get(k) == Some(v))
            && self.tombstones.is_subset(&other.tombstones);
        let other_sub_self = other
            .entries
            .iter()
            .all(|(k, v)| self.entries.get(k) == Some(v))
            && other.tombstones.is_subset(&self.tombstones);
        match (self_sub_other, other_sub_self) {
            (true, true) => Some(Ordering::Equal),
            (true, false) => Some(Ordering::Less),
            (false, true) => Some(Ordering::Greater),
            (false, false) => None,
        }
    }
}

impl<A: Ord + Clone> JoinSemilattice for OrSet<A> {
    /// Union of entries and union of tombstones.
    ///
    /// For shared tags, the values must be equal (invariant: a tag
    /// uniquely identifies one operation).  Self's value is kept
    /// for determinism.
    fn join(&self, other: &Self) -> Self {
        let from_self = self.entries.iter().map(|(k, v)| (*k, v.clone()));
        let only_in_other = other
            .entries
            .iter()
            .filter(|(k, _)| !self.entries.contains_key(k))
            .map(|(k, v)| (*k, v.clone()));
        Self {
            entries: from_self.chain(only_in_other).collect(),
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

    #[test]
    fn empty_set_has_no_elements() {
        let s: OrSet<u64> = OrSet::empty();
        assert!(s.is_empty());
        assert_eq!(s.len(), 0);
    }

    #[test]
    fn add_makes_element_visible() {
        let s = OrSet::empty().add(42_u64, r(0), t(1));
        assert!(s.contains(&42));
        assert_eq!(s.len(), 1);
    }

    #[test]
    fn remove_makes_element_invisible() {
        let s = OrSet::empty()
            .add(42_u64, r(0), t(1))
            .remove(&42);
        assert!(!s.contains(&42));
        assert!(s.is_empty());
    }

    #[test]
    fn add_after_remove_resurrects_element() {
        let s = OrSet::empty()
            .add(42_u64, r(0), t(1))
            .remove(&42)
            .add(42, r(0), t(2));
        assert!(s.contains(&42));
    }

    #[test]
    fn concurrent_add_survives_remote_remove() {
        // Replica 0 has {42} tagged with t(1)
        let base = OrSet::empty().add(42_u64, r(0), t(1));

        // Replica 0 removes 42 (tombstones tag t(1))
        let removed = base.remove(&42);

        // Replica 1 concurrently adds 42 with a new tag
        let added = base.add(42, r(1), t(2));

        // Merge: tag t(2) is NOT tombstoned, so 42 is visible
        let merged = removed.join(&added);
        assert!(merged.contains(&42));
    }

    #[test]
    fn join_merges_disjoint_elements() {
        let a = OrSet::empty().add(1_u64, r(0), t(1));
        let b = OrSet::empty().add(2_u64, r(1), t(1));
        let merged = a.join(&b);
        assert!(merged.contains(&1));
        assert!(merged.contains(&2));
        assert_eq!(merged.len(), 2);
    }

    #[test]
    fn join_merges_tombstones() {
        let base = OrSet::empty().add(42_u64, r(0), t(1));
        let removed_a = base.remove(&42);
        let removed_b = base.remove(&42);
        let merged = removed_a.join(&removed_b);
        assert!(!merged.contains(&42));
    }

    #[test]
    fn multiple_tags_for_same_element() {
        let s = OrSet::empty()
            .add(42_u64, r(0), t(1))
            .add(42, r(1), t(2));
        assert!(s.contains(&42));
        assert_eq!(s.len(), 1);

        // Removing only tombstones both tags
        let removed = s.remove(&42);
        assert!(!removed.contains(&42));
    }

    #[derive(Debug)]
    enum Op {
        Add(u64, u64),
        Remove(u64),
    }

    /// Derive an element deterministically from a tag's components.
    ///
    /// This ensures two independently-generated `OrSet`s that share
    /// a tag always agree on the element, preserving the invariant
    /// that a tag uniquely identifies one operation.
    fn element_for_tag(rid: u64, ts: u64) -> u64 {
        (rid.wrapping_mul(31).wrapping_add(ts)) % 5
    }

    fn arb_orset() -> impl Strategy<Value = OrSet<u64>> {
        proptest::collection::vec(
            prop_oneof![
                (0u64..3, 0u64..20)
                    .prop_map(|(rid, ts)| Op::Add(rid, ts)),
                (0u64..5).prop_map(Op::Remove),
            ],
            0..8,
        )
        .prop_map(|ops| {
            ops.into_iter()
                .fold(OrSet::empty(), |set, op| match op {
                    Op::Add(rid, ts) => {
                        set.add(
                            element_for_tag(rid, ts),
                            ReplicaId::new(rid),
                            Timestamp::new(ts),
                        )
                    }
                    Op::Remove(e) => set.remove(&e),
                })
        })
    }

    proptest! {
        #[test]
        fn semilattice_join_is_commutative(
            a in arb_orset(),
            b in arb_orset(),
        ) {
            prop_assert_eq!(a.join(&b), b.join(&a));
        }

        #[test]
        fn semilattice_join_is_associative(
            a in arb_orset(),
            b in arb_orset(),
            c in arb_orset(),
        ) {
            prop_assert_eq!(a.join(&b).join(&c), a.join(&b.join(&c)));
        }

        #[test]
        fn semilattice_join_is_idempotent(a in arb_orset()) {
            prop_assert_eq!(a.join(&a), a);
        }

        #[test]
        fn semilattice_join_is_upper_bound(
            a in arb_orset(),
            b in arb_orset(),
        ) {
            let joined = a.join(&b);
            prop_assert!(a <= joined);
            prop_assert!(b <= joined);
        }

        #[test]
        fn add_is_monotonic(
            a in arb_orset(),
            replica in 0u64..3,
            ts in 0u64..100,
            elem in 0u64..5,
        ) {
            let b = a.add(elem, ReplicaId::new(replica), Timestamp::new(ts));
            prop_assert!(a <= b);
        }

        #[test]
        fn remove_is_monotonic(a in arb_orset(), elem in 0u64..5) {
            let b = a.remove(&elem);
            prop_assert!(a <= b);
        }
    }
}
