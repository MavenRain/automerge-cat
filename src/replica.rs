//! Domain primitives for distributed replica identity and time.

use serde::{Deserialize, Serialize};

/// A replica identifier: uniquely identifies a node in the distributed system.
///
/// Newtype over `u64` to prevent confusion with timestamps, counts, etc.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ReplicaId(u64);

impl ReplicaId {
    /// Create a new replica identifier.
    #[must_use]
    pub fn new(id: u64) -> Self {
        Self(id)
    }

    /// The underlying identifier.
    #[must_use]
    pub fn value(self) -> u64 {
        self.0
    }
}

impl core::fmt::Display for ReplicaId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "replica:{}", self.0)
    }
}

/// A logical timestamp for causal ordering.
///
/// Newtype over `u64`.  Timestamps are totally ordered; ties in
/// CRDT merge operations are broken by [`ReplicaId`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Timestamp(u64);

impl Timestamp {
    /// Create a new timestamp.
    #[must_use]
    pub fn new(t: u64) -> Self {
        Self(t)
    }

    /// The underlying value.
    #[must_use]
    pub fn value(self) -> u64 {
        self.0
    }

    /// Advance the timestamp by one tick.
    #[must_use]
    pub fn tick(self) -> Self {
        Self(self.0 + 1)
    }
}

impl core::fmt::Display for Timestamp {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "t:{}", self.0)
    }
}

/// A unique causal tag identifying a specific operation by a specific replica.
///
/// Used in observed-remove CRDTs to track individual additions
/// so they can be precisely tombstoned on removal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Tag(ReplicaId, Timestamp);

impl Tag {
    /// Create a new tag.
    #[must_use]
    pub fn new(replica: ReplicaId, timestamp: Timestamp) -> Self {
        Self(replica, timestamp)
    }

    /// The replica that produced this tag.
    #[must_use]
    pub fn replica(self) -> ReplicaId {
        self.0
    }

    /// The timestamp of this tag.
    #[must_use]
    pub fn timestamp(self) -> Timestamp {
        self.1
    }
}

impl core::fmt::Display for Tag {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}@{}", self.0, self.1)
    }
}
