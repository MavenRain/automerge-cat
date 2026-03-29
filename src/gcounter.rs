//! Grow-only counter: the simplest state-based CRDT.
//!
//! Each replica maintains its own monotonically increasing count.
//! The total value is the sum of all replica counts.  The merge
//! (join) is pointwise max.
//!
//! The state space forms a join-semilattice where:
//! - Partial order: pointwise `<=` on the count map
//! - Join (coproduct): pointwise `max`
//!
//! By `comp_cat_rs::collapse::join_is_colimit`, this join is a
//! left Kan extension.

use std::collections::{BTreeMap, BTreeSet};

use comp_cat_rs::foundation::JoinSemilattice;
use serde::{Deserialize, Serialize};
use crate::replica::ReplicaId;

/// A grow-only counter (`GCounter`).
///
/// Internally a map from [`ReplicaId`] to `u64`.  Entries with
/// count zero are never stored; absence means zero.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GCounter {
    counts: BTreeMap<ReplicaId, u64>,
}

impl GCounter {
    /// An empty counter: all replicas at zero.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            counts: BTreeMap::new(),
        }
    }

    /// Create a counter from raw entries.
    ///
    /// Entries with count zero are silently dropped.  If multiple
    /// entries share a [`ReplicaId`], the last one wins.
    #[must_use]
    pub fn from_entries(entries: impl IntoIterator<Item = (ReplicaId, u64)>) -> Self {
        Self {
            counts: entries.into_iter().filter(|(_, v)| *v > 0).collect(),
        }
    }

    /// Increment the count for a replica, returning the new counter.
    ///
    /// This is the only write operation on a `GCounter`.  Each replica
    /// should only increment its own count.
    #[must_use]
    pub fn increment(&self, replica: ReplicaId) -> Self {
        let current = self.counts.get(&replica).copied().unwrap_or(0);
        Self {
            counts: self
                .counts
                .iter()
                .map(|(k, v)| (*k, *v))
                .filter(|(k, _)| *k != replica)
                .chain(core::iter::once((replica, current + 1)))
                .collect(),
        }
    }

    /// The total count across all replicas.
    #[must_use]
    pub fn value(&self) -> u64 {
        self.counts.values().sum()
    }

    /// The count for a specific replica.
    #[must_use]
    pub fn count_for(&self, replica: ReplicaId) -> u64 {
        self.counts.get(&replica).copied().unwrap_or(0)
    }

    /// The number of replicas that have incremented.
    #[must_use]
    pub fn replica_count(&self) -> usize {
        self.counts.len()
    }
}

impl PartialEq for GCounter {
    fn eq(&self, other: &Self) -> bool {
        self.counts == other.counts
    }
}

impl Eq for GCounter {}

impl PartialOrd for GCounter {
    /// Pointwise comparison: `a <= b` iff for every replica `r`,
    /// `a[r] <= b[r]`.
    ///
    /// Two counters are incomparable when one has a higher count
    /// for some replica and a lower count for another.
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        use core::cmp::Ordering;
        self.counts
            .keys()
            .chain(other.counts.keys())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .try_fold(Ordering::Equal, |ord, k| {
                let left = self.counts.get(k).copied().unwrap_or(0);
                let right = other.counts.get(k).copied().unwrap_or(0);
                match (ord, left.cmp(&right)) {
                    (Ordering::Equal, cmp) => Some(cmp),
                    (_, Ordering::Equal)
                    | (Ordering::Less, Ordering::Less)
                    | (Ordering::Greater, Ordering::Greater) => Some(ord),
                    _ => None,
                }
            })
    }
}

impl JoinSemilattice for GCounter {
    /// Pointwise max of the count maps.
    ///
    /// This is the coproduct in the posetal category of `GCounter` states.
    fn join(&self, other: &Self) -> Self {
        let from_self = self.counts.iter().map(|(k, v)| {
            let right = other.counts.get(k).copied().unwrap_or(0);
            (*k, (*v).max(right))
        });
        let only_in_other = other
            .counts
            .iter()
            .map(|(k, v)| (*k, *v))
            .filter(|(k, _)| !self.counts.contains_key(k));
        Self {
            counts: from_self.chain(only_in_other).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::cmp::Ordering;
    use proptest::prelude::*;

    fn r(id: u64) -> ReplicaId {
        ReplicaId::new(id)
    }

    #[test]
    fn empty_counter_has_zero_value() {
        assert_eq!(GCounter::empty().value(), 0);
    }

    #[test]
    fn single_increment_produces_value_one() {
        let c = GCounter::empty().increment(r(0));
        assert_eq!(c.value(), 1);
        assert_eq!(c.count_for(r(0)), 1);
    }

    #[test]
    fn multiple_increments_accumulate() {
        let c = GCounter::empty()
            .increment(r(0))
            .increment(r(0))
            .increment(r(1));
        assert_eq!(c.value(), 3);
        assert_eq!(c.count_for(r(0)), 2);
        assert_eq!(c.count_for(r(1)), 1);
    }

    #[test]
    fn join_of_disjoint_replicas_sums_both() {
        let a = GCounter::empty().increment(r(0)).increment(r(0));
        let b = GCounter::empty().increment(r(1));
        let merged = a.join(&b);
        assert_eq!(merged.value(), 3);
        assert_eq!(merged.count_for(r(0)), 2);
        assert_eq!(merged.count_for(r(1)), 1);
    }

    #[test]
    fn join_of_overlapping_replicas_takes_max() {
        let a = GCounter::from_entries(vec![(r(0), 3), (r(1), 1)]);
        let b = GCounter::from_entries(vec![(r(0), 1), (r(1), 5)]);
        let merged = a.join(&b);
        assert_eq!(merged.count_for(r(0)), 3);
        assert_eq!(merged.count_for(r(1)), 5);
        assert_eq!(merged.value(), 8);
    }

    #[test]
    fn diverged_replicas_are_incomparable() {
        let a = GCounter::from_entries(vec![(r(0), 3), (r(1), 1)]);
        let b = GCounter::from_entries(vec![(r(0), 1), (r(1), 5)]);
        assert_eq!(a.partial_cmp(&b), None);
    }

    #[test]
    fn dominated_counter_is_less() {
        let a = GCounter::from_entries(vec![(r(0), 1), (r(1), 2)]);
        let b = GCounter::from_entries(vec![(r(0), 3), (r(1), 5)]);
        assert_eq!(a.partial_cmp(&b), Some(Ordering::Less));
        assert_eq!(b.partial_cmp(&a), Some(Ordering::Greater));
    }

    #[test]
    fn equal_counters_compare_equal() {
        let a = GCounter::from_entries(vec![(r(0), 2), (r(1), 3)]);
        let b = GCounter::from_entries(vec![(r(0), 2), (r(1), 3)]);
        assert_eq!(a.partial_cmp(&b), Some(Ordering::Equal));
        assert_eq!(a, b);
    }

    #[test]
    fn increment_is_monotonic() {
        let a = GCounter::empty().increment(r(0)).increment(r(1));
        let b = a.increment(r(0));
        assert!(a <= b);
        assert!(a != b);
    }

    fn arb_gcounter() -> impl Strategy<Value = GCounter> {
        proptest::collection::vec((0u64..5, 1u64..20), 0..5).prop_map(|entries| {
            GCounter::from_entries(
                entries
                    .into_iter()
                    .map(|(rid, c)| (ReplicaId::new(rid), c)),
            )
        })
    }

    proptest! {
        #[test]
        fn semilattice_join_is_commutative(
            a in arb_gcounter(),
            b in arb_gcounter(),
        ) {
            prop_assert_eq!(a.join(&b), b.join(&a));
        }

        #[test]
        fn semilattice_join_is_associative(
            a in arb_gcounter(),
            b in arb_gcounter(),
            c in arb_gcounter(),
        ) {
            prop_assert_eq!(a.join(&b).join(&c), a.join(&b.join(&c)));
        }

        #[test]
        fn semilattice_join_is_idempotent(a in arb_gcounter()) {
            prop_assert_eq!(a.join(&a), a);
        }

        #[test]
        fn semilattice_join_is_upper_bound(
            a in arb_gcounter(),
            b in arb_gcounter(),
        ) {
            let joined = a.join(&b);
            prop_assert!(a <= joined);
            prop_assert!(b <= joined);
        }

        #[test]
        fn increment_grows_monotonically(
            a in arb_gcounter(),
            replica in 0u64..5,
        ) {
            let b = a.increment(ReplicaId::new(replica));
            prop_assert!(a <= b);
        }
    }
}
