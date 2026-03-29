//! Positive-negative counter: supports both increment and decrement.
//!
//! Implemented as a product of two [`GCounter`]s (positive and
//! negative).  The observed value is `positive - negative`.
//!
//! The state space forms a join-semilattice as the categorical
//! product of two `GCounter` semilattices: the partial order is
//! componentwise, and the join is componentwise.

use comp_cat_rs::foundation::JoinSemilattice;
use serde::{Deserialize, Serialize};
use crate::gcounter::GCounter;
use crate::replica::ReplicaId;

/// A positive-negative counter (`PNCounter`).
///
/// Internally a pair of [`GCounter`]s.  The positive counter
/// tracks increments; the negative counter tracks decrements.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PnCounter {
    pos: GCounter,
    neg: GCounter,
}

impl PnCounter {
    /// A zero counter.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            pos: GCounter::empty(),
            neg: GCounter::empty(),
        }
    }

    /// Create a counter from its positive and negative components.
    #[must_use]
    pub fn from_components(pos: GCounter, neg: GCounter) -> Self {
        Self { pos, neg }
    }

    /// Increment (add one), returning the new counter.
    #[must_use]
    pub fn increment(&self, replica: ReplicaId) -> Self {
        Self {
            pos: self.pos.increment(replica),
            neg: self.neg.clone(),
        }
    }

    /// Decrement (subtract one), returning the new counter.
    #[must_use]
    pub fn decrement(&self, replica: ReplicaId) -> Self {
        Self {
            pos: self.pos.clone(),
            neg: self.neg.increment(replica),
        }
    }

    /// The observed value: positive count minus negative count.
    #[must_use]
    pub fn value(&self) -> i128 {
        i128::from(self.pos.value()) - i128::from(self.neg.value())
    }

    /// The positive component.
    #[must_use]
    pub fn positive(&self) -> &GCounter {
        &self.pos
    }

    /// The negative component.
    #[must_use]
    pub fn negative(&self) -> &GCounter {
        &self.neg
    }
}

impl PartialEq for PnCounter {
    fn eq(&self, other: &Self) -> bool {
        self.pos == other.pos && self.neg == other.neg
    }
}

impl Eq for PnCounter {}

impl PartialOrd for PnCounter {
    /// Product order: `a <= b` iff `a.pos <= b.pos` AND `a.neg <= b.neg`.
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        use core::cmp::Ordering;
        match (
            self.pos.partial_cmp(&other.pos),
            self.neg.partial_cmp(&other.neg),
        ) {
            (Some(Ordering::Equal), cmp) | (cmp, Some(Ordering::Equal)) => cmp,
            (Some(Ordering::Less), Some(Ordering::Less)) => Some(Ordering::Less),
            (Some(Ordering::Greater), Some(Ordering::Greater)) => Some(Ordering::Greater),
            _ => None,
        }
    }
}

impl JoinSemilattice for PnCounter {
    /// Componentwise join of the positive and negative counters.
    fn join(&self, other: &Self) -> Self {
        Self {
            pos: self.pos.join(&other.pos),
            neg: self.neg.join(&other.neg),
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
        assert_eq!(PnCounter::empty().value(), 0);
    }

    #[test]
    fn increment_increases_value() {
        let c = PnCounter::empty().increment(r(0)).increment(r(0));
        assert_eq!(c.value(), 2);
    }

    #[test]
    fn decrement_decreases_value() {
        let c = PnCounter::empty().decrement(r(0));
        assert_eq!(c.value(), -1);
    }

    #[test]
    fn mixed_operations_net_correctly() {
        let c = PnCounter::empty()
            .increment(r(0))
            .increment(r(0))
            .increment(r(1))
            .decrement(r(0))
            .decrement(r(1));
        assert_eq!(c.value(), 1);
    }

    #[test]
    fn join_merges_both_components() {
        let a = PnCounter::empty().increment(r(0)).increment(r(0));
        let b = PnCounter::empty().decrement(r(1));
        let merged = a.join(&b);
        assert_eq!(merged.value(), 1);
    }

    #[test]
    fn diverged_counters_are_incomparable() {
        let a = PnCounter::empty().increment(r(0));
        let b = PnCounter::empty().decrement(r(0));
        assert_eq!(a.partial_cmp(&b), None);
    }

    #[test]
    fn dominated_counter_is_less() {
        let a = PnCounter::empty().increment(r(0));
        let b = a.increment(r(0)).decrement(r(1));
        assert_eq!(a.partial_cmp(&b), Some(Ordering::Less));
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

    fn arb_pncounter() -> impl Strategy<Value = PnCounter> {
        (arb_gcounter(), arb_gcounter())
            .prop_map(|(pos, neg)| PnCounter::from_components(pos, neg))
    }

    proptest! {
        #[test]
        fn semilattice_join_is_commutative(
            a in arb_pncounter(),
            b in arb_pncounter(),
        ) {
            prop_assert_eq!(a.join(&b), b.join(&a));
        }

        #[test]
        fn semilattice_join_is_associative(
            a in arb_pncounter(),
            b in arb_pncounter(),
            c in arb_pncounter(),
        ) {
            prop_assert_eq!(a.join(&b).join(&c), a.join(&b.join(&c)));
        }

        #[test]
        fn semilattice_join_is_idempotent(a in arb_pncounter()) {
            prop_assert_eq!(a.join(&a), a);
        }

        #[test]
        fn semilattice_join_is_upper_bound(
            a in arb_pncounter(),
            b in arb_pncounter(),
        ) {
            let joined = a.join(&b);
            prop_assert!(a <= joined);
            prop_assert!(b <= joined);
        }

        #[test]
        fn increment_grows_monotonically(
            a in arb_pncounter(),
            replica in 0u64..5,
        ) {
            prop_assert!(a <= a.increment(ReplicaId::new(replica)));
        }

        #[test]
        fn decrement_grows_monotonically(
            a in arb_pncounter(),
            replica in 0u64..5,
        ) {
            prop_assert!(a <= a.decrement(ReplicaId::new(replica)));
        }
    }
}
