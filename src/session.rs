//! Session: ergonomic builder for `Document` and `OpLog`.
//!
//! Wraps a `Document`, `OpLog`, `ReplicaId`, and an auto-incrementing
//! `Timestamp` so callers don't have to thread identity and clock
//! through every operation.

use std::collections::BTreeSet;
use crate::document::{Document, NodeId, Value};
use crate::error::Error;
use crate::oplog::{Action, OpLog};
use crate::replica::{ReplicaId, Tag, Timestamp};
use crate::rga::Origin;

/// A local editing session on a document.
///
/// Each session belongs to one replica and auto-increments its
/// logical clock.  Every mutation updates both the `Document`
/// (state) and the `OpLog` (history) in lockstep.
#[derive(Debug, Clone)]
pub struct Session {
    doc: Document,
    log: OpLog,
    replica: ReplicaId,
    clock: u64,
}

impl Session {
    /// Start a new session on an empty document.
    #[must_use]
    pub fn new(replica: ReplicaId) -> Self {
        Self {
            doc: Document::new(),
            log: OpLog::empty(),
            replica,
            clock: 0,
        }
    }

    /// Start a session from an existing document and log.
    ///
    /// The clock is set to one past the highest timestamp in the
    /// log to avoid tag collisions.
    #[must_use]
    pub fn from_state(doc: Document, log: OpLog, replica: ReplicaId) -> Self {
        let clock = log
            .tags()
            .iter()
            .map(|t| t.timestamp().value())
            .max()
            .map_or(0, |m| m + 1);
        Self {
            doc,
            log,
            replica,
            clock,
        }
    }

    /// The current document state.
    #[must_use]
    pub fn document(&self) -> &Document {
        &self.doc
    }

    /// The operation log.
    #[must_use]
    pub fn log(&self) -> &OpLog {
        &self.log
    }

    /// This session's replica ID.
    #[must_use]
    pub fn replica(&self) -> ReplicaId {
        self.replica
    }

    /// The current logical clock value as a `Timestamp`.
    #[must_use]
    pub fn clock(&self) -> Timestamp {
        Timestamp::new(self.clock)
    }

    /// The raw clock counter (used by the sync module).
    #[must_use]
    pub fn clock_value(&self) -> u64 {
        self.clock
    }

    /// Construct a session from pre-built parts (used by the sync module).
    #[must_use]
    pub fn from_parts(doc: Document, log: OpLog, replica: ReplicaId, clock: u64) -> Self {
        Self { doc, log, replica, clock }
    }

    // -- map operations -----------------------------------------------------

    /// Set a key in a map node.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NodeNotFound`] or [`Error::WrongNodeType`]
    /// if the node is missing or not a map.
    pub fn set_key(
        &self,
        node: NodeId,
        key: &str,
        value: &Value,
    ) -> Result<Self, Error> {
        let ts = self.next_timestamp();
        let doc = self.doc.set_key(node, key, value, self.replica, ts)?;
        let log = self.log.append(
            Action::SetKey {
                node,
                key: key.to_string(),
                value: value.clone(),
            },
            self.replica,
            ts,
        )?;
        Ok(self.advance(doc, log))
    }

    /// Delete a key from a map node.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NodeNotFound`] or [`Error::WrongNodeType`]
    /// if the node is missing or not a map.
    pub fn delete_key(
        &self,
        node: NodeId,
        key: &str,
    ) -> Result<Self, Error> {
        let ts = self.next_timestamp();
        let doc = self.doc.delete_key(node, key)?;
        let log = self.log.append(
            Action::DeleteKey {
                node,
                key: key.to_string(),
            },
            self.replica,
            ts,
        )?;
        Ok(self.advance(doc, log))
    }

    // -- list operations ----------------------------------------------------

    /// Insert a value into a list node.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NodeNotFound`] or [`Error::WrongNodeType`]
    /// if the node is missing or not a list.
    pub fn list_insert(
        &self,
        node: NodeId,
        origin: Origin,
        value: Value,
    ) -> Result<Self, Error> {
        let ts = self.next_timestamp();
        let doc = self.doc.list_insert(
            node,
            origin,
            value.clone(),
            self.replica,
            ts,
        )?;
        let log = self.log.append(
            Action::ListInsert {
                node,
                origin,
                value,
            },
            self.replica,
            ts,
        )?;
        Ok(self.advance(doc, log))
    }

    /// Delete an element from a list node by tag.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NodeNotFound`] or [`Error::WrongNodeType`]
    /// if the node is missing or not a list.
    pub fn list_delete(
        &self,
        node: NodeId,
        element_tag: Tag,
    ) -> Result<Self, Error> {
        let ts = self.next_timestamp();
        let doc = self.doc.list_delete(node, element_tag)?;
        let log = self.log.append(
            Action::ListDelete {
                node,
                element_tag,
            },
            self.replica,
            ts,
        )?;
        Ok(self.advance(doc, log))
    }

    // -- container creation -------------------------------------------------

    /// Create a new empty map container.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NodeAlreadyExists`] on tag collision.
    pub fn create_map(&self) -> Result<(Self, NodeId), Error> {
        let ts = self.next_timestamp();
        let (doc, id) = self.doc.create_map(self.replica, ts)?;
        let log = self.log.append(Action::CreateMap, self.replica, ts)?;
        Ok((self.advance(doc, log), id))
    }

    /// Create a new empty list container.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NodeAlreadyExists`] on tag collision.
    pub fn create_list(&self) -> Result<(Self, NodeId), Error> {
        let ts = self.next_timestamp();
        let (doc, id) = self.doc.create_list(self.replica, ts)?;
        let log = self.log.append(Action::CreateList, self.replica, ts)?;
        Ok((self.advance(doc, log), id))
    }

    // -- compaction ---------------------------------------------------------

    /// Compact the document and log, removing tombstoned entries
    /// and operations that all replicas have observed.
    ///
    /// `safe_tags` should be the intersection of all known peers'
    /// tag sets: tags that every replica has seen.  Use
    /// [`crate::sync::Peer`] tracking to compute this.
    #[must_use]
    pub fn compact(&self, safe_tags: &BTreeSet<Tag>) -> Self {
        Self {
            doc: self.doc.compact(safe_tags),
            log: self.log.compact(safe_tags),
            replica: self.replica,
            clock: self.clock,
        }
    }

    // -- merge --------------------------------------------------------------

    /// Merge another session's log into this one.
    ///
    /// The merged document is re-materialized from the joined logs.
    #[must_use]
    pub fn merge(&self, other: &Session) -> Self {
        use comp_cat_rs::foundation::JoinSemilattice;
        let log = self.log.join(&other.log);
        let doc = log.materialize();
        let clock = log
            .tags()
            .iter()
            .map(|t| t.timestamp().value())
            .max()
            .map_or(self.clock, |m| m + 1);
        Self {
            doc,
            log,
            replica: self.replica,
            clock,
        }
    }

    // -- internal -----------------------------------------------------------

    fn next_timestamp(&self) -> Timestamp {
        Timestamp::new(self.clock)
    }

    fn advance(&self, doc: Document, log: OpLog) -> Self {
        Self {
            doc,
            log,
            replica: self.replica,
            clock: self.clock + 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn r(id: u64) -> ReplicaId {
        ReplicaId::new(id)
    }

    #[test]
    fn basic_set_and_get() -> Result<(), Error> {
        let s = Session::new(r(0))
            .set_key(NodeId::Root, "name", &Value::Str("alice".into()))?;
        let vals = s.document().get_key(NodeId::Root, "name")?;
        assert!(vals.contains(&Value::Str("alice".into())));
        Ok(())
    }

    #[test]
    fn clock_auto_increments() -> Result<(), Error> {
        let s = Session::new(r(0));
        assert_eq!(s.clock(), Timestamp::new(0));
        let s = s.set_key(NodeId::Root, "a", &Value::Int(1))?;
        assert_eq!(s.clock(), Timestamp::new(1));
        let s = s.set_key(NodeId::Root, "b", &Value::Int(2))?;
        assert_eq!(s.clock(), Timestamp::new(2));
        Ok(())
    }

    #[test]
    fn oplog_tracks_operations() -> Result<(), Error> {
        let s = Session::new(r(0))
            .set_key(NodeId::Root, "x", &Value::Int(1))?
            .set_key(NodeId::Root, "y", &Value::Int(2))?;
        assert_eq!(s.log().len(), 2);
        Ok(())
    }

    #[test]
    fn nested_containers() -> Result<(), Error> {
        let s = Session::new(r(0));
        let (s, child) = s.create_map()?;
        let s = s
            .set_key(NodeId::Root, "child", &Value::Map(child))?
            .set_key(child, "inner", &Value::Int(42))?;
        let vals = s.document().get_key(child, "inner")?;
        assert!(vals.contains(&Value::Int(42)));
        Ok(())
    }

    #[test]
    fn list_operations() -> Result<(), Error> {
        let s = Session::new(r(0));
        let (s, list) = s.create_list()?;
        let s = s
            .set_key(NodeId::Root, "items", &Value::List(list))?
            .list_insert(list, Origin::Head, Value::Str("a".into()))?
            .list_insert(list, Origin::Head, Value::Str("b".into()))?;
        let elems = s.document().list_elements(list)?;
        assert_eq!(elems, vec![&Value::Str("b".into()), &Value::Str("a".into())]);
        Ok(())
    }

    #[test]
    fn merge_two_sessions() -> Result<(), Error> {
        let a = Session::new(r(0))
            .set_key(NodeId::Root, "x", &Value::Int(1))?;
        let b = Session::new(r(1))
            .set_key(NodeId::Root, "y", &Value::Int(2))?;
        let merged = a.merge(&b);
        assert!(merged.document().get_key(NodeId::Root, "x")?.contains(&Value::Int(1)));
        assert!(merged.document().get_key(NodeId::Root, "y")?.contains(&Value::Int(2)));
        Ok(())
    }

    #[test]
    fn concurrent_edits_via_merge() -> Result<(), Error> {
        let base = Session::new(r(0))
            .set_key(NodeId::Root, "x", &Value::Int(0))?;
        let a = Session::from_state(
            base.document().clone(),
            base.log().clone(),
            r(0),
        )
        .set_key(NodeId::Root, "x", &Value::Int(1))?;
        let b = Session::from_state(
            base.document().clone(),
            base.log().clone(),
            r(1),
        )
        .set_key(NodeId::Root, "x", &Value::Int(2))?;
        let merged = a.merge(&b);
        let vals = merged.document().get_key(NodeId::Root, "x")?;
        assert_eq!(vals.len(), 2);
        Ok(())
    }

    #[test]
    fn materialized_state_matches_document() -> Result<(), Error> {
        let s = Session::new(r(0))
            .set_key(NodeId::Root, "a", &Value::Int(1))?
            .set_key(NodeId::Root, "b", &Value::Int(2))?
            .set_key(NodeId::Root, "a", &Value::Int(3))?;
        assert_eq!(s.log().materialize(), *s.document());
        Ok(())
    }

    #[test]
    fn compact_preserves_observable_state() -> Result<(), Error> {
        // Write x=1 then x=2; the first write is tombstoned.
        let s = Session::new(r(0))
            .set_key(NodeId::Root, "x", &Value::Int(1))?
            .set_key(NodeId::Root, "x", &Value::Int(2))?;

        // Compact all tags (pretend all replicas have seen everything).
        let all_tags = s.log().tags();
        let compacted = s.compact(&all_tags);

        // Observable state is unchanged.
        let vals = compacted.document().get_key(NodeId::Root, "x")?;
        assert_eq!(vals.len(), 1);
        assert!(vals.contains(&Value::Int(2)));

        // But the log is smaller.
        assert!(compacted.log().len() < s.log().len());
        Ok(())
    }

    #[test]
    fn compact_shrinks_tombstones() {
        use crate::orset::OrSet;

        let set = OrSet::empty()
            .add(1_u64, r(0), Timestamp::new(1))
            .add(2, r(0), Timestamp::new(2))
            .remove(&1);

        assert!(set.contains(&2));
        assert!(!set.contains(&1));

        let all: BTreeSet<Tag> = [
            Tag::new(r(0), Timestamp::new(1)),
            Tag::new(r(0), Timestamp::new(2)),
        ]
        .into_iter()
        .collect();
        let compacted = set.compact(&all);

        assert!(compacted.contains(&2));
        assert!(!compacted.contains(&1));
    }
}
