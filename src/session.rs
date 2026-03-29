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

    // -- text operations ----------------------------------------------------

    /// Insert a character into a text (list) node at a visible index.
    ///
    /// Index 0 inserts at the beginning.  An index equal to the
    /// node's element count appends at the end.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NodeNotFound`] or [`Error::WrongNodeType`]
    /// if the node is missing or not a list.
    pub fn text_insert(
        &self,
        node: NodeId,
        index: usize,
        ch: char,
    ) -> Result<Self, Error> {
        let ts = self.next_timestamp();
        let origin = self
            .doc
            .list_entries(node)?
            .get(index.wrapping_sub(1))
            .filter(|_| index > 0)
            .map_or(Origin::Head, |(tag, _)| Origin::After(*tag));
        let doc = self.doc.list_insert(
            node,
            origin,
            Value::Str(ch.to_string()),
            self.replica,
            ts,
        )?;
        let log = self.log.append(
            Action::TextInsert { node, index, ch },
            self.replica,
            ts,
        )?;
        Ok(self.advance(doc, log))
    }

    /// Delete the character at a visible index in a text (list) node.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NodeNotFound`], [`Error::WrongNodeType`],
    /// or [`Error::IndexOutOfBounds`].
    pub fn text_delete(
        &self,
        node: NodeId,
        index: usize,
    ) -> Result<Self, Error> {
        let ts = self.next_timestamp();
        let entries = self.doc.list_entries(node)?;
        let tag = entries
            .get(index)
            .map(|(tag, _)| *tag)
            .ok_or(Error::IndexOutOfBounds {
                index,
                len: entries.len(),
            })?;
        let doc = self.doc.list_delete(node, tag)?;
        let log = self.log.append(
            Action::TextDelete { node, index },
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

    // -- text helpers -------------------------------------------------------

    /// Create a text container, attach it to a map key, and
    /// populate it with an initial string.
    ///
    /// Returns the updated session and the text node's ID.
    /// Each character consumes one clock tick, so this is safe
    /// to interleave with other Session operations (no timestamp
    /// collisions).
    ///
    /// # Errors
    ///
    /// Propagates errors from container creation, key setting,
    /// and character insertion.
    pub fn create_text(
        &self,
        parent: NodeId,
        key: &str,
        content: &str,
    ) -> Result<(Self, NodeId), Error> {
        let (session, text_id) = self.create_list()?;
        let session = session.set_key(parent, key, &Value::Text(text_id))?;
        let session = content
            .chars()
            .enumerate()
            .try_fold(session, |s, (i, ch)| s.text_insert(text_id, i, ch))?;
        Ok((session, text_id))
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

    /// Merge another session's state into this one.
    ///
    /// Joins the documents directly (state-based `CvRDT` merge) and
    /// joins the logs (for history).  This is O(nodes + ops) rather
    /// than O(ops^2) for full re-materialization.
    #[must_use]
    pub fn merge(&self, other: &Session) -> Self {
        use comp_cat_rs::foundation::JoinSemilattice;
        let doc = self.doc.join(&other.doc);
        let log = self.log.join(&other.log);
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
    fn merge_matches_rematerialization() -> Result<(), Error> {
        let base = Session::new(r(0))
            .set_key(NodeId::Root, "x", &Value::Int(0))?;
        let a = Session::from_state(
            base.document().clone(),
            base.log().clone(),
            r(0),
        )
        .set_key(NodeId::Root, "x", &Value::Int(1))?
        .set_key(NodeId::Root, "y", &Value::Int(10))?;
        let b = Session::from_state(
            base.document().clone(),
            base.log().clone(),
            r(1),
        )
        .set_key(NodeId::Root, "x", &Value::Int(2))?
        .set_key(NodeId::Root, "z", &Value::Int(20))?;
        let merged = a.merge(&b);
        let from_log = merged.log().materialize();
        assert_eq!(*merged.document(), from_log);
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

    #[test]
    fn text_insert_builds_string() -> Result<(), Error> {
        let s = Session::new(r(0));
        let (s, text) = s.create_list()?;
        let s = s
            .set_key(NodeId::Root, "content", &Value::Text(text))?
            .text_insert(text, 0, 'h')?
            .text_insert(text, 1, 'i')?;
        let elems = s.document().list_elements(text)?;
        assert_eq!(elems.len(), 2);
        assert_eq!(elems[0], &Value::Str("h".into()));
        assert_eq!(elems[1], &Value::Str("i".into()));
        Ok(())
    }

    #[test]
    fn text_delete_removes_character() -> Result<(), Error> {
        let s = Session::new(r(0));
        let (s, text) = s.create_list()?;
        let s = s
            .text_insert(text, 0, 'a')?
            .text_insert(text, 1, 'b')?
            .text_insert(text, 2, 'c')?
            .text_delete(text, 1)?;
        let elems = s.document().list_elements(text)?;
        assert_eq!(elems.len(), 2);
        assert_eq!(elems[0], &Value::Str("a".into()));
        assert_eq!(elems[1], &Value::Str("c".into()));
        Ok(())
    }

    #[test]
    fn text_delete_out_of_bounds_returns_error() -> Result<(), Error> {
        let s = Session::new(r(0));
        let (s, text) = s.create_list()?;
        let s = s.text_insert(text, 0, 'a')?;
        let result = s.text_delete(text, 5);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn text_ops_flow_through_oplog() -> Result<(), Error> {
        let s = Session::new(r(0));
        let (s, text) = s.create_list()?;
        let s = s
            .text_insert(text, 0, 'a')?
            .text_insert(text, 1, 'b')?
            .text_delete(text, 0)?;
        // 3 text ops + 1 create_list + 1 (implicit from create_list in session)
        assert!(s.log().len() >= 3);
        // Materialize matches document
        assert_eq!(s.log().materialize(), *s.document());
        Ok(())
    }

    #[test]
    fn create_text_builds_string_safely() -> Result<(), Error> {
        let s = Session::new(r(0));
        let (s, text_id) = s.create_text(NodeId::Root, "greeting", "hello")?;

        // Text is readable
        let elems = s.document().list_elements(text_id)?;
        assert_eq!(elems.len(), 5);

        // Key is set
        assert!(s.document().get_key(NodeId::Root, "greeting")?.contains(&Value::Text(text_id)));

        // Further operations don't collide
        let s = s
            .set_key(NodeId::Root, "other", &Value::Int(42))?
            .text_insert(text_id, 5, '!');
        assert!(s.is_ok());

        // Materialize matches document
        let s = s?;
        assert_eq!(s.log().materialize(), *s.document());
        Ok(())
    }
}
