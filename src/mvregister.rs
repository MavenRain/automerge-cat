//! Multi-value register: a CRDT that preserves concurrent writes.
//!
//! Unlike [`LwwRegister`](crate::lww_register::LwwRegister), which
//! picks a single winner, `MvRegister` keeps all concurrently-written
//! values.  A subsequent write on a merged state tombstones all
//! concurrent values, resolving the conflict.
//!
//! Structurally identical to [`OrSet`](crate::orset::OrSet): a
//! grow-only map from [`Tag`] to value, plus a grow-only set of
//! tombstoned [`Tag`]s.  The difference is the API: `write`
//! tombstones all currently-observed tags before adding the new
//! entry, whereas `OrSet::add` does not.
//!
//! The state space forms a join-semilattice as a product of two
//! grow-only structures (entries + tombstones), with componentwise
//! subset ordering and componentwise union as the join.

use std::collections::{BTreeMap, BTreeSet};

use comp_cat_rs::foundation::JoinSemilattice;
use serde::{Deserialize, Serialize};
use crate::replica::{ReplicaId, Tag, Timestamp};

/// A multi-value register (`MvRegister`).
///
/// Observed values are entries whose tags have not been tombstoned.
/// Concurrent writes produce multiple observed values; a subsequent
/// write resolves the conflict.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MvRegister<A: Ord> {
    entries: BTreeMap<Tag, A>,
    tombstones: BTreeSet<Tag>,
}

impl<A: Ord> MvRegister<A> {
    /// An empty register: no value has been written.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            entries: BTreeMap::new(),
            tombstones: BTreeSet::new(),
        }
    }

    /// A register with an initial value.
    #[must_use]
    pub fn new(value: A, replica: ReplicaId, timestamp: Timestamp) -> Self {
        Self {
            entries: core::iter::once((Tag::new(replica, timestamp), value)).collect(),
            tombstones: BTreeSet::new(),
        }
    }

    /// Write a new value, tombstoning all currently-observed values.
    ///
    /// If the tag already exists, the write is a no-op (tags must
    /// be globally unique).
    #[must_use]
    pub fn write(
        &self,
        value: A,
        replica: ReplicaId,
        timestamp: Timestamp,
    ) -> Self
    where
        A: Clone,
    {
        let tag = Tag::new(replica, timestamp);
        if self.entries.contains_key(&tag) {
            self.clone()
        } else {
            let new_tombs: BTreeSet<Tag> = self
                .entries
                .keys()
                .filter(|t| !self.tombstones.contains(t))
                .copied()
                .collect();
            Self {
                entries: self
                    .entries
                    .iter()
                    .map(|(k, v)| (*k, v.clone()))
                    .chain(core::iter::once((tag, value)))
                    .collect(),
                tombstones: self
                    .tombstones
                    .union(&new_tombs)
                    .copied()
                    .collect(),
            }
        }
    }

    /// Tombstone all currently-observed values without writing a new one.
    ///
    /// After clear, the register is empty.  A concurrent write from
    /// another replica will survive the merge (add-wins semantics).
    #[must_use]
    pub fn clear(&self) -> Self
    where
        A: Clone,
    {
        let new_tombs: BTreeSet<Tag> = self
            .entries
            .keys()
            .filter(|t| !self.tombstones.contains(t))
            .copied()
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
    ///
    /// For each tag in `safe_tags`, if the tag is both in `entries`
    /// and `tombstones`, both are removed.  This shrinks the state
    /// without changing the observable values.
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

    /// All currently-observed values (non-tombstoned).
    ///
    /// Returns one value when the register is resolved, multiple
    /// values when there are unresolved concurrent writes, or an
    /// empty set when nothing has been written.
    #[must_use]
    pub fn values(&self) -> BTreeSet<&A> {
        self.entries
            .iter()
            .filter(|(tag, _)| !self.tombstones.contains(tag))
            .map(|(_, v)| v)
            .collect()
    }

    /// Whether exactly one value is observed (no unresolved concurrency).
    #[must_use]
    pub fn is_resolved(&self) -> bool {
        self.values().len() == 1
    }

    /// Whether no value has been written or all writes are tombstoned.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries
            .keys()
            .all(|tag| self.tombstones.contains(tag))
    }
}

impl<A: Ord> PartialEq for MvRegister<A> {
    fn eq(&self, other: &Self) -> bool {
        self.entries == other.entries && self.tombstones == other.tombstones
    }
}

impl<A: Ord> Eq for MvRegister<A> {}

impl<A: Ord> PartialOrd for MvRegister<A> {
    /// Product of subset orders on entries and tombstones.
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

impl<A: Ord + Clone> JoinSemilattice for MvRegister<A> {
    /// Union of entries and union of tombstones.
    ///
    /// For shared tags, values must be equal (invariant: a tag
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

    fn element_for_tag(rid: u64, ts: u64) -> u64 {
        (rid.wrapping_mul(31).wrapping_add(ts)) % 5
    }

    #[test]
    fn new_register_has_one_value() {
        let reg = MvRegister::new(42_u64, r(0), t(1));
        assert!(reg.is_resolved());
        assert!(reg.values().contains(&42));
    }

    #[test]
    fn sequential_write_replaces_value() {
        let reg = MvRegister::new(1_u64, r(0), t(1))
            .write(2, r(0), t(2));
        assert!(reg.is_resolved());
        assert!(reg.values().contains(&2));
        assert!(!reg.values().contains(&1));
    }

    #[test]
    fn concurrent_writes_produce_multiple_values() {
        let base = MvRegister::new(1_u64, r(0), t(1));
        let a = base.write(2, r(0), t(2));
        let b = base.write(3, r(1), t(2));
        let merged = a.join(&b);
        assert!(merged.values().contains(&2));
        assert!(merged.values().contains(&3));
        assert!(!merged.values().contains(&1));
        assert_eq!(merged.values().len(), 2);
        assert!(!merged.is_resolved());
    }

    #[test]
    fn write_after_merge_resolves_concurrency() {
        let base = MvRegister::new(1_u64, r(0), t(1));
        let a = base.write(2, r(0), t(2));
        let b = base.write(3, r(1), t(2));
        let merged = a.join(&b);
        let resolved = merged.write(4, r(0), t(3));
        assert!(resolved.is_resolved());
        assert!(resolved.values().contains(&4));
    }

    #[test]
    fn empty_register_has_no_values() {
        let reg: MvRegister<u64> = MvRegister::empty();
        assert!(reg.is_empty());
        assert!(reg.values().is_empty());
    }

    #[test]
    fn write_on_empty_produces_one_value() {
        let reg = MvRegister::<u64>::empty().write(7, r(0), t(1));
        assert!(reg.is_resolved());
        assert!(reg.values().contains(&7));
    }

    fn arb_mvregister() -> impl Strategy<Value = MvRegister<u64>> {
        proptest::collection::vec((0u64..3, 0u64..20), 0..6).prop_map(|writes| {
            writes.iter().fold(MvRegister::empty(), |reg, &(rid, ts)| {
                reg.write(
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
            a in arb_mvregister(),
            b in arb_mvregister(),
        ) {
            prop_assert_eq!(a.join(&b), b.join(&a));
        }

        #[test]
        fn semilattice_join_is_associative(
            a in arb_mvregister(),
            b in arb_mvregister(),
            c in arb_mvregister(),
        ) {
            prop_assert_eq!(a.join(&b).join(&c), a.join(&b.join(&c)));
        }

        #[test]
        fn semilattice_join_is_idempotent(a in arb_mvregister()) {
            prop_assert_eq!(a.join(&a), a);
        }

        #[test]
        fn semilattice_join_is_upper_bound(
            a in arb_mvregister(),
            b in arb_mvregister(),
        ) {
            let joined = a.join(&b);
            prop_assert!(a <= joined);
            prop_assert!(b <= joined);
        }

        #[test]
        fn write_is_monotonic(
            a in arb_mvregister(),
            rid in 0u64..3,
            ts in 0u64..100,
        ) {
            let b = a.write(
                element_for_tag(rid, ts),
                ReplicaId::new(rid),
                Timestamp::new(ts),
            );
            prop_assert!(a <= b);
        }
    }
}
