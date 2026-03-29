//! Sync: replica-to-replica operation exchange protocol.
//!
//! Provides the wire types and peer-tracking state for incremental
//! synchronization.  A [`Peer`] tracks which operations a remote
//! replica is known to have; [`Peer::generate_message`] produces a
//! [`SyncMessage`] containing the ops the remote is missing;
//! [`Session::receive_sync`] integrates incoming changes.
//!
//! ## Protocol
//!
//! ```text
//! Alice                          Bob
//!   |                              |
//!   |-- generate_message(log) ---->|  (changes Bob is missing)
//!   |                              |-- receive_sync(msg) (integrate)
//!   |                              |
//!   |<-- generate_message(log) ---|  (changes Alice is missing)
//!   |-- receive_sync(msg) ------  |
//!   |                              |
//!   |  (both converged)            |
//! ```
//!
//! After a full round-trip, both replicas hold the same `OpLog`
//! (up to commutativity of join).

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use crate::oplog::OpLog;
use crate::replica::Tag;

/// A bundle of changes to send to a remote replica.
///
/// Serializable for transmission over any transport (TCP, WebSocket,
/// file, etc.) via serde.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SyncMessage {
    changes: Vec<Change>,
}

impl SyncMessage {
    /// The changes in this message.
    #[must_use]
    pub fn changes(&self) -> &[Change] {
        &self.changes
    }

    /// The number of changes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.changes.len()
    }

    /// Whether the message carries no changes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.changes.is_empty()
    }

    /// The tags of all operations in this message.
    #[must_use]
    pub fn tags(&self) -> BTreeSet<Tag> {
        self.changes.iter().map(|c| c.tag).collect()
    }
}

/// A single change: an operation plus its causal dependencies.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Change {
    tag: Tag,
    action: crate::oplog::Action,
    deps: BTreeSet<Tag>,
}

impl Change {
    /// The tag identifying this operation.
    #[must_use]
    pub fn tag(&self) -> Tag {
        self.tag
    }

    /// The action this operation performs.
    #[must_use]
    pub fn action(&self) -> &crate::oplog::Action {
        &self.action
    }

    /// The causal dependencies of this operation.
    #[must_use]
    pub fn deps(&self) -> &BTreeSet<Tag> {
        &self.deps
    }
}

/// Per-remote-peer sync state.
///
/// Tracks which operation tags a remote replica is known to have,
/// so [`generate_message`](Peer::generate_message) sends only
/// what the remote is missing.
#[derive(Debug, Clone)]
pub struct Peer {
    their_tags: BTreeSet<Tag>,
}

impl Default for Peer {
    fn default() -> Self {
        Self::new()
    }
}

impl Peer {
    /// A new peer with no known state.
    #[must_use]
    pub fn new() -> Self {
        Self {
            their_tags: BTreeSet::new(),
        }
    }

    /// Generate a message containing ops the peer is missing.
    #[must_use]
    pub fn generate_message(&self, local_log: &OpLog) -> SyncMessage {
        let changes = local_log
            .tags()
            .iter()
            .filter(|t| !self.their_tags.contains(t))
            .filter_map(|t| {
                local_log.get(*t).map(|op| Change {
                    tag: op.tag(),
                    action: op.action().clone(),
                    deps: local_log.dependencies(*t),
                })
            })
            .collect();
        SyncMessage { changes }
    }

    /// Update this peer after sending a message.
    ///
    /// Records that the remote now has these operations.
    #[must_use]
    pub fn record_sent(&self, msg: &SyncMessage) -> Self {
        Self {
            their_tags: self
                .their_tags
                .iter()
                .copied()
                .chain(msg.changes.iter().map(|c| c.tag))
                .collect(),
        }
    }

    /// Update this peer after receiving a message from them.
    ///
    /// Records that the remote has at least these operations.
    #[must_use]
    pub fn record_received(&self, msg: &SyncMessage) -> Self {
        Self {
            their_tags: self
                .their_tags
                .iter()
                .copied()
                .chain(msg.changes.iter().map(|c| c.tag))
                .collect(),
        }
    }

    /// The tags we believe the remote has.
    #[must_use]
    pub fn their_tags(&self) -> &BTreeSet<Tag> {
        &self.their_tags
    }

    /// Whether we believe the peer is fully up-to-date with the
    /// given log (no ops to send).
    #[must_use]
    pub fn is_synced(&self, local_log: &OpLog) -> bool {
        local_log
            .tags()
            .iter()
            .all(|t| self.their_tags.contains(t))
    }
}

/// Apply a [`SyncMessage`] to a [`Session`], integrating the
/// remote's changes.
///
/// Returns the updated session with the new ops merged in.
impl crate::session::Session {
    /// Integrate changes from a remote replica.
    ///
    /// Each change is appended to the local `OpLog` with its
    /// original dependencies, then the document is re-materialized.
    #[must_use]
    pub fn receive_sync(&self, msg: &SyncMessage) -> Self {
        let log = msg.changes.iter().fold(self.log().clone(), |log, change| {
            log.append_with_deps(
                change.action.clone(),
                change.tag.replica(),
                change.tag.timestamp(),
                change.deps.clone(),
            )
            .unwrap_or(log)
        });
        let doc = log.materialize();
        let clock = log
            .tags()
            .iter()
            .map(|t| t.timestamp().value())
            .max()
            .map_or(self.clock_value(), |m| m + 1);
        Self::from_parts(doc, log, self.replica(), clock)
    }
}

/// Compute the set of tags safe to compact: tags that ALL given
/// peers have observed AND that are in the local log.
///
/// Pass the `their_tags()` of every known peer.  The result is the
/// intersection of all peer tag sets, intersected with the local
/// log's tags.
#[must_use]
pub fn safe_compaction_tags(
    local_log: &OpLog,
    peers: &[&Peer],
) -> BTreeSet<Tag> {
    let local = local_log.tags();
    peers
        .split_first()
        .map_or_else(BTreeSet::new, |(first, rest)| {
            rest.iter().fold(
                first
                    .their_tags()
                    .intersection(&local)
                    .copied()
                    .collect::<BTreeSet<Tag>>(),
                |acc, peer| {
                    acc.intersection(peer.their_tags())
                        .copied()
                        .collect()
                },
            )
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::{NodeId, Value};
    use crate::error::Error;
    use crate::replica::ReplicaId;
    use crate::session::Session;

    fn r(id: u64) -> ReplicaId {
        ReplicaId::new(id)
    }

    #[test]
    fn sync_transfers_ops() -> Result<(), Error> {
        let alice = Session::new(r(0))
            .set_key(NodeId::Root, "x", &Value::Int(1))?;
        let bob = Session::new(r(1));

        let peer = Peer::new();
        let msg = peer.generate_message(alice.log());
        assert_eq!(msg.len(), 1);

        let bob = bob.receive_sync(&msg);
        assert!(bob.document().get_key(NodeId::Root, "x")?.contains(&Value::Int(1)));
        Ok(())
    }

    #[test]
    fn bidirectional_sync_converges() -> Result<(), Error> {
        let alice = Session::new(r(0))
            .set_key(NodeId::Root, "x", &Value::Int(1))?;
        let bob = Session::new(r(1))
            .set_key(NodeId::Root, "y", &Value::Int(2))?;

        // Alice -> Bob
        let alice_peer = Peer::new();
        let msg_to_bob = alice_peer.generate_message(alice.log());
        let bob = bob.receive_sync(&msg_to_bob);

        // Bob -> Alice
        let bob_peer = Peer::new();
        let msg_to_alice = bob_peer.generate_message(bob.log());
        let alice = alice.receive_sync(&msg_to_alice);

        // Both have x and y
        assert!(alice.document().get_key(NodeId::Root, "x")?.contains(&Value::Int(1)));
        assert!(alice.document().get_key(NodeId::Root, "y")?.contains(&Value::Int(2)));
        assert!(bob.document().get_key(NodeId::Root, "x")?.contains(&Value::Int(1)));
        assert!(bob.document().get_key(NodeId::Root, "y")?.contains(&Value::Int(2)));
        Ok(())
    }

    #[test]
    fn concurrent_edits_merge_via_sync() -> Result<(), Error> {
        let base = Session::new(r(0))
            .set_key(NodeId::Root, "x", &Value::Int(0))?;

        let alice = Session::from_state(
            base.document().clone(),
            base.log().clone(),
            r(0),
        )
        .set_key(NodeId::Root, "x", &Value::Int(1))?;

        let bob = Session::from_state(
            base.document().clone(),
            base.log().clone(),
            r(1),
        )
        .set_key(NodeId::Root, "x", &Value::Int(2))?;

        // Full round-trip
        let msg_a = Peer::new().generate_message(alice.log());
        let msg_b = Peer::new().generate_message(bob.log());
        let alice = alice.receive_sync(&msg_b);
        let bob = bob.receive_sync(&msg_a);

        // Both see the conflict
        let a_vals = alice.document().get_key(NodeId::Root, "x")?;
        let b_vals = bob.document().get_key(NodeId::Root, "x")?;
        assert_eq!(a_vals.len(), 2);
        assert_eq!(a_vals, b_vals);
        Ok(())
    }

    #[test]
    fn peer_tracking_avoids_resending() -> Result<(), Error> {
        let alice = Session::new(r(0))
            .set_key(NodeId::Root, "x", &Value::Int(1))?;

        let peer = Peer::new();
        let msg = peer.generate_message(alice.log());
        assert_eq!(msg.len(), 1);

        let peer = peer.record_sent(&msg);
        let msg2 = peer.generate_message(alice.log());
        assert!(msg2.is_empty());
        Ok(())
    }

    #[test]
    fn is_synced_reports_correctly() -> Result<(), Error> {
        let alice = Session::new(r(0))
            .set_key(NodeId::Root, "x", &Value::Int(1))?;

        let peer = Peer::new();
        assert!(!peer.is_synced(alice.log()));

        let msg = peer.generate_message(alice.log());
        let peer = peer.record_sent(&msg);
        assert!(peer.is_synced(alice.log()));
        Ok(())
    }

    #[test]
    fn sync_message_round_trips_via_bincode() -> Result<(), Box<dyn std::error::Error>> {
        let alice = Session::new(r(0))
            .set_key(NodeId::Root, "x", &Value::Int(1))?
            .set_key(NodeId::Root, "y", &Value::Str("hello".into()))?;
        let msg = Peer::new().generate_message(alice.log());

        let bytes = bincode::serialize(&msg)?;
        let msg2: SyncMessage = bincode::deserialize(&bytes)?;
        assert_eq!(msg, msg2);
        Ok(())
    }

    #[test]
    fn three_way_sync() -> Result<(), Error> {
        let alice = Session::new(r(0))
            .set_key(NodeId::Root, "a", &Value::Int(1))?;
        let bob = Session::new(r(1))
            .set_key(NodeId::Root, "b", &Value::Int(2))?;
        let carol = Session::new(r(2))
            .set_key(NodeId::Root, "c", &Value::Int(3))?;

        // Alice syncs with Bob
        let msg = Peer::new().generate_message(alice.log());
        let bob = bob.receive_sync(&msg);
        let msg = Peer::new().generate_message(bob.log());
        let alice = alice.receive_sync(&msg);

        // Alice (who now has a+b) syncs with Carol
        let msg = Peer::new().generate_message(alice.log());
        let carol = carol.receive_sync(&msg);
        let msg = Peer::new().generate_message(carol.log());
        let alice = alice.receive_sync(&msg);

        // Alice has all three
        assert!(alice.document().get_key(NodeId::Root, "a")?.contains(&Value::Int(1)));
        assert!(alice.document().get_key(NodeId::Root, "b")?.contains(&Value::Int(2)));
        assert!(alice.document().get_key(NodeId::Root, "c")?.contains(&Value::Int(3)));
        Ok(())
    }

    #[test]
    fn empty_sync_produces_no_changes() {
        let alice = Session::new(r(0));
        let msg = Peer::new().generate_message(alice.log());
        assert!(msg.is_empty());
    }

    #[test]
    fn safe_compaction_after_full_sync() -> Result<(), Error> {
        let alice = Session::new(r(0))
            .set_key(NodeId::Root, "x", &Value::Int(1))?
            .set_key(NodeId::Root, "x", &Value::Int(2))?;
        let bob = Session::new(r(1));

        // Full bidirectional sync
        let mut alice_peer = Peer::new();
        let msg = alice_peer.generate_message(alice.log());
        alice_peer = alice_peer.record_sent(&msg);
        let bob = bob.receive_sync(&msg);
        let mut bob_peer = Peer::new().record_received(&msg);

        let msg = bob_peer.generate_message(bob.log());
        bob_peer = bob_peer.record_sent(&msg);
        alice_peer = alice_peer.record_received(&msg);
        let alice = alice.receive_sync(&msg);

        // Both fully synced; all tags are safe to compact
        let safe = safe_compaction_tags(alice.log(), &[&alice_peer, &bob_peer]);
        assert!(!safe.is_empty());

        let alice = alice.compact(&safe);
        let bob = bob.compact(&safe);

        // Observable state preserved
        assert!(alice.document().get_key(NodeId::Root, "x")?.contains(&Value::Int(2)));
        assert!(bob.document().get_key(NodeId::Root, "x")?.contains(&Value::Int(2)));

        // Logs are smaller
        assert!(alice.log().len() < 2);
        Ok(())
    }
}
