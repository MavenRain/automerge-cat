//! Last-writer-wins register: a CRDT storing a single value.
//!
//! The merge (join) takes the value with the later timestamp.
//! Ties are broken deterministically by [`ReplicaId`].
//!
//! The state space forms a (total-order) join-semilattice where:
//! - Total order: `(timestamp, replica_id)` lexicographic
//! - Join (coproduct): `max` by this order
//!
//! Because the order is total, every pair of states is comparable.
//! The coproduct in a totally ordered set is simply the maximum.
//!
//! By `comp_cat_rs::collapse::join_is_colimit`, this join is a
//! left Kan extension.

use core::cmp::Ordering;

use comp_cat_rs::foundation::JoinSemilattice;
use serde::{Deserialize, Serialize};
use crate::replica::{ReplicaId, Timestamp};

/// A last-writer-wins register storing a value of type `A`.
///
/// The ordering is on `(timestamp, replica_id)`.  The value `A`
/// does not participate in the ordering; it is carried along with
/// the winning metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LwwRegister<A> {
    value: A,
    timestamp: Timestamp,
    replica: ReplicaId,
}

impl<A> LwwRegister<A> {
    /// Create a new register with an initial value.
    #[must_use]
    pub fn new(value: A, timestamp: Timestamp, replica: ReplicaId) -> Self {
        Self {
            value,
            timestamp,
            replica,
        }
    }

    /// The current value.
    #[must_use]
    pub fn value(&self) -> &A {
        &self.value
    }

    /// The timestamp of the current value.
    #[must_use]
    pub fn timestamp(&self) -> Timestamp {
        self.timestamp
    }

    /// The replica that wrote the current value.
    #[must_use]
    pub fn replica(&self) -> ReplicaId {
        self.replica
    }

    /// Write a new value, returning the updated register.
    ///
    /// If the new write is causally older than the current state
    /// (lower timestamp, or same timestamp with lower replica ID),
    /// the current state is retained.  This is equivalent to
    /// merging with a single-write register via `join`.
    #[must_use]
    pub fn write(
        &self,
        value: A,
        timestamp: Timestamp,
        replica: ReplicaId,
    ) -> Self
    where
        A: Clone,
    {
        let candidate = Self::new(value, timestamp, replica);
        self.join(&candidate)
    }

    /// Compare the metadata (timestamp, replica) of two registers.
    fn cmp_metadata(&self, other: &Self) -> Ordering {
        (self.timestamp, self.replica).cmp(&(other.timestamp, other.replica))
    }
}

impl<A> PartialEq for LwwRegister<A> {
    /// Two registers are equal iff they have the same metadata.
    ///
    /// A single replica does not produce two different values at
    /// the same logical time, so equal metadata implies equal value.
    fn eq(&self, other: &Self) -> bool {
        self.timestamp == other.timestamp && self.replica == other.replica
    }
}

impl<A> Eq for LwwRegister<A> {}

impl<A> PartialOrd for LwwRegister<A> {
    /// Lexicographic order on `(timestamp, replica_id)`.
    ///
    /// This is a total order, so `partial_cmp` always returns `Some`.
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp_metadata(other))
    }
}

impl<A: Clone> JoinSemilattice for LwwRegister<A> {
    /// Take the register with the later `(timestamp, replica_id)`.
    ///
    /// This is the coproduct in the totally ordered posetal category:
    /// simply `max(a, b)`.
    fn join(&self, other: &Self) -> Self {
        match self.cmp_metadata(other) {
            Ordering::Less => other.clone(),
            Ordering::Equal | Ordering::Greater => self.clone(),
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
    fn later_timestamp_wins() {
        let a = LwwRegister::new("old", t(1), r(0));
        let b = LwwRegister::new("new", t(2), r(0));
        let merged = a.join(&b);
        assert_eq!(*merged.value(), "new");
        assert_eq!(merged.timestamp(), t(2));
    }

    #[test]
    fn higher_replica_breaks_tie() {
        let a = LwwRegister::new("from_r0", t(5), r(0));
        let b = LwwRegister::new("from_r1", t(5), r(1));
        let merged = a.join(&b);
        assert_eq!(*merged.value(), "from_r1");
        assert_eq!(merged.replica(), r(1));
    }

    #[test]
    fn write_with_newer_timestamp_updates() {
        let reg = LwwRegister::new("first", t(1), r(0));
        let updated = reg.write("second", t(2), r(0));
        assert_eq!(*updated.value(), "second");
    }

    #[test]
    fn write_with_older_timestamp_is_ignored() {
        let reg = LwwRegister::new("current", t(5), r(0));
        let updated = reg.write("stale", t(3), r(1));
        assert_eq!(*updated.value(), "current");
    }

    #[test]
    fn total_order_means_always_comparable() {
        let a = LwwRegister::new(1_u64, t(3), r(0));
        let b = LwwRegister::new(2_u64, t(7), r(1));
        assert!(a.partial_cmp(&b).is_some());
    }

    fn arb_lww_register() -> impl Strategy<Value = LwwRegister<u64>> {
        (0u64..100, 0u64..5, 0u64..1000).prop_map(|(ts, replica, value)| {
            LwwRegister::new(value, Timestamp::new(ts), ReplicaId::new(replica))
        })
    }

    proptest! {
        #[test]
        fn semilattice_join_is_commutative(
            a in arb_lww_register(),
            b in arb_lww_register(),
        ) {
            prop_assert_eq!(a.join(&b), b.join(&a));
        }

        #[test]
        fn semilattice_join_is_associative(
            a in arb_lww_register(),
            b in arb_lww_register(),
            c in arb_lww_register(),
        ) {
            prop_assert_eq!(a.join(&b).join(&c), a.join(&b.join(&c)));
        }

        #[test]
        fn semilattice_join_is_idempotent(a in arb_lww_register()) {
            prop_assert_eq!(a.join(&a), a);
        }

        #[test]
        fn semilattice_join_is_upper_bound(
            a in arb_lww_register(),
            b in arb_lww_register(),
        ) {
            let joined = a.join(&b);
            prop_assert!(a <= joined);
            prop_assert!(b <= joined);
        }
    }
}
