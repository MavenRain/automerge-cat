//! Text: a string CRDT built on [`Rga<char>`].
//!
//! Provides an index-based API for collaborative text editing:
//! insert characters at a position, delete a range, read the
//! string.  Internally translates between visible character
//! indices and the tag-based addressing of the underlying
//! [`Rga`].
//!
//! The semilattice structure delegates entirely to `Rga<char>`.

use comp_cat_rs::foundation::JoinSemilattice;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use crate::error::Error;
use crate::replica::{ReplicaId, Tag, Timestamp};
use crate::rga::{Origin, Rga};

/// A collaborative text CRDT.
///
/// Wraps [`Rga<char>`] with a string-level API.  Characters are
/// addressed by visible index (0-based, counting only non-tombstoned
/// characters).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Text {
    rga: Rga<char>,
}

impl Text {
    /// An empty text.
    #[must_use]
    pub fn empty() -> Self {
        Self { rga: Rga::empty() }
    }

    /// Build a text from an iterator of `(char, ReplicaId, Timestamp)`.
    ///
    /// Uses [`Rga::from_sequence`] for O(n log n) bulk construction.
    #[must_use]
    pub fn from_chars(
        chars: impl IntoIterator<Item = (char, ReplicaId, Timestamp)>,
    ) -> Self {
        Self {
            rga: Rga::from_sequence(chars),
        }
    }

    /// Create a text from an initial string.
    ///
    /// Each character gets a unique tag derived from the replica
    /// and a base timestamp incremented per character.
    ///
    /// **Warning:** the generated timestamps may collide with a
    /// [`Session`](crate::session::Session)'s auto-clock.  Prefer
    /// [`Session::create_text`](crate::session::Session::create_text)
    /// when working within a session.  This method is intended for
    /// standalone `Text` usage outside of a session.
    ///
    /// # Errors
    ///
    /// Returns [`Error::IndexOutOfBounds`] if internal index
    /// arithmetic overflows (should not happen in practice).
    pub fn from_str_with(
        s: &str,
        replica: ReplicaId,
        base_timestamp: Timestamp,
    ) -> Result<Self, Error> {
        s.chars()
            .enumerate()
            .try_fold(Self::empty(), |text, (i, ch)| {
                let ts = Timestamp::new(base_timestamp.value() + i as u64);
                text.insert(text.len(), ch, replica, ts)
            })
    }

    /// Insert a character at the given visible index.
    ///
    /// Index 0 inserts at the beginning.  An index equal to
    /// [`len`](Self::len) appends at the end.
    ///
    /// # Errors
    ///
    /// Returns [`Error::IndexOutOfBounds`] if `index > len`.
    pub fn insert(
        &self,
        index: usize,
        ch: char,
        replica: ReplicaId,
        timestamp: Timestamp,
    ) -> Result<Self, Error> {
        let len = self.len();
        if index > len {
            Err(Error::IndexOutOfBounds { index, len })
        } else {
            let origin = self.origin_at(index);
            Ok(Self {
                rga: self.rga.insert_after(origin, ch, replica, timestamp),
            })
        }
    }

    /// Insert a string at the given visible index.
    ///
    /// Each character gets a unique tag: the base timestamp is
    /// incremented by the character's offset within the string.
    ///
    /// # Errors
    ///
    /// Returns [`Error::IndexOutOfBounds`] if `index > len`.
    pub fn insert_str(
        &self,
        index: usize,
        s: &str,
        replica: ReplicaId,
        base_timestamp: Timestamp,
    ) -> Result<Self, Error> {
        s.chars()
            .enumerate()
            .try_fold(self.clone(), |text, (i, ch)| {
                let ts = Timestamp::new(base_timestamp.value() + i as u64);
                text.insert(index + i, ch, replica, ts)
            })
    }

    /// Delete the character at the given visible index.
    ///
    /// # Errors
    ///
    /// Returns [`Error::IndexOutOfBounds`] if `index >= len`.
    pub fn delete_at(&self, index: usize) -> Result<Self, Error> {
        self.tag_at(index)
            .map(|tag| Self {
                rga: self.rga.delete(tag),
            })
            .ok_or(Error::IndexOutOfBounds {
                index,
                len: self.len(),
            })
    }

    /// Delete a range of characters `[start..end)`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::IndexOutOfBounds`] if `start > len`.
    pub fn delete_range(&self, start: usize, end: usize) -> Result<Self, Error> {
        let len = self.len();
        if start > len {
            Err(Error::IndexOutOfBounds { index: start, len })
        } else {
            let clamped_end = end.min(len);
            let tags: Vec<Tag> = self
                .rga
                .entries()
                .into_iter()
                .skip(start)
                .take(clamped_end.saturating_sub(start))
                .map(|(tag, _)| tag)
                .collect();
            Ok(tags.iter().fold(self.clone(), |text, tag| Self {
                rga: text.rga.delete(*tag),
            }))
        }
    }

    /// Compact the underlying RGA, removing tombstoned entries
    /// that all replicas have observed.
    #[must_use]
    pub fn compact(&self, safe_tags: &BTreeSet<Tag>) -> Self {
        Self {
            rga: self.rga.compact(safe_tags),
        }
    }

    /// The number of visible characters.
    #[must_use]
    pub fn len(&self) -> usize {
        self.rga.len()
    }

    /// Whether the text is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rga.is_empty()
    }

    /// The character at the given visible index.
    #[must_use]
    pub fn char_at(&self, index: usize) -> Option<char> {
        self.rga
            .elements()
            .into_iter()
            .nth(index)
            .copied()
    }

    /// The visible characters as a `Vec`.
    #[must_use]
    pub fn chars(&self) -> Vec<char> {
        self.rga.elements().into_iter().copied().collect()
    }

    /// The underlying [`Rga`], for advanced tag-level operations.
    #[must_use]
    pub fn as_rga(&self) -> &Rga<char> {
        &self.rga
    }

    /// Find the [`Origin`] for inserting at a visible index.
    ///
    /// Index 0 → `Origin::Head`.
    /// Index `n` → `Origin::After(tag_of_nth_visible_char - 1)`.
    fn origin_at(&self, index: usize) -> Origin {
        let entries = self.rga.entries();
        if index == 0 {
            Origin::Head
        } else {
            entries
                .get(index - 1)
                .map_or_else(
                    || entries.last().map_or(Origin::Head, |(tag, _)| Origin::After(*tag)),
                    |(tag, _)| Origin::After(*tag),
                )
        }
    }

    /// Find the [`Tag`] of the character at a visible index.
    fn tag_at(&self, index: usize) -> Option<Tag> {
        self.rga
            .entries()
            .into_iter()
            .nth(index)
            .map(|(tag, _)| tag)
    }
}

// ---------------------------------------------------------------------------
// Semilattice (delegates to Rga<char>)
// ---------------------------------------------------------------------------

impl PartialEq for Text {
    fn eq(&self, other: &Self) -> bool {
        self.rga == other.rga
    }
}

impl Eq for Text {}

impl PartialOrd for Text {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        self.rga.partial_cmp(&other.rga)
    }
}

impl JoinSemilattice for Text {
    fn join(&self, other: &Self) -> Self {
        Self {
            rga: self.rga.join(&other.rga),
        }
    }
}

impl core::fmt::Display for Text {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.rga
            .elements()
            .iter()
            .try_for_each(|ch| write!(f, "{ch}"))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

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
    fn empty_text_has_zero_length() {
        let text = Text::empty();
        assert!(text.is_empty());
        assert_eq!(text.len(), 0);
        assert_eq!(text.to_string(), "");
    }

    #[test]
    fn insert_single_character() -> Result<(), Error> {
        let text = Text::empty().insert(0, 'a', r(0), t(1))?;
        assert_eq!(text.to_string(), "a");
        assert_eq!(text.len(), 1);
        Ok(())
    }

    #[test]
    fn insert_builds_string_left_to_right() -> Result<(), Error> {
        let text = Text::empty()
            .insert(0, 'a', r(0), t(1))?
            .insert(1, 'b', r(0), t(2))?
            .insert(2, 'c', r(0), t(3))?;
        assert_eq!(text.to_string(), "abc");
        Ok(())
    }

    #[test]
    fn insert_at_beginning() -> Result<(), Error> {
        let text = Text::empty()
            .insert(0, 'b', r(0), t(1))?
            .insert(0, 'a', r(0), t(2))?;
        assert_eq!(text.to_string(), "ab");
        Ok(())
    }

    #[test]
    fn insert_in_middle() -> Result<(), Error> {
        let text = Text::empty()
            .insert(0, 'a', r(0), t(1))?
            .insert(1, 'c', r(0), t(2))?
            .insert(1, 'b', r(0), t(3))?;
        assert_eq!(text.to_string(), "abc");
        Ok(())
    }

    #[test]
    fn from_str_with_creates_text() -> Result<(), Error> {
        let text = Text::from_str_with("hello", r(0), t(1))?;
        assert_eq!(text.to_string(), "hello");
        assert_eq!(text.len(), 5);
        Ok(())
    }

    #[test]
    fn insert_str_at_index() -> Result<(), Error> {
        let text = Text::from_str_with("hd", r(0), t(1))?
            .insert_str(1, "ello worl", r(0), t(100))?;
        assert_eq!(text.to_string(), "hello world");
        Ok(())
    }

    #[test]
    fn delete_at_removes_character() -> Result<(), Error> {
        let text = Text::from_str_with("abc", r(0), t(1))?.delete_at(1)?;
        assert_eq!(text.to_string(), "ac");
        Ok(())
    }

    #[test]
    fn delete_range_removes_characters() -> Result<(), Error> {
        let text = Text::from_str_with("hello world", r(0), t(1))?
            .delete_range(5, 11)?;
        assert_eq!(text.to_string(), "hello");
        Ok(())
    }

    #[test]
    fn delete_out_of_bounds_returns_error() -> Result<(), Error> {
        let text = Text::from_str_with("abc", r(0), t(1))?;
        assert!(text.delete_at(10).is_err());
        Ok(())
    }

    #[test]
    fn insert_out_of_bounds_returns_error() {
        let text = Text::empty();
        assert!(text.insert(5, 'x', r(0), t(1)).is_err());
    }

    #[test]
    fn char_at_returns_correct_character() -> Result<(), Error> {
        let text = Text::from_str_with("abc", r(0), t(1))?;
        assert_eq!(text.char_at(0), Some('a'));
        assert_eq!(text.char_at(1), Some('b'));
        assert_eq!(text.char_at(2), Some('c'));
        assert_eq!(text.char_at(3), None);
        Ok(())
    }

    #[test]
    fn concurrent_inserts_at_same_position() -> Result<(), Error> {
        let base = Text::from_str_with("ac", r(0), t(1))?;
        let left = base.insert(1, 'L', r(0), t(10))?;
        let right = base.insert(1, 'R', r(1), t(10))?;
        let merged = left.join(&right);
        let s = merged.to_string();
        assert_eq!(s.len(), 4);
        assert!(s.starts_with('a'));
        assert!(s.ends_with('c'));
        assert!(s.contains('L'));
        assert!(s.contains('R'));
        Ok(())
    }

    #[test]
    fn concurrent_insert_and_delete() -> Result<(), Error> {
        let base = Text::from_str_with("abc", r(0), t(1))?;
        let deleted = base.delete_at(1)?;
        let b_tag = base.as_rga().entries()[1].0;
        let inserted = Text {
            rga: base.as_rga().insert_after(
                Origin::After(b_tag),
                'X',
                r(1),
                t(10),
            ),
        };
        let merged = deleted.join(&inserted);
        let s = merged.to_string();
        assert!(s.contains('X'));
        assert!(!s.contains('b'));
        Ok(())
    }

    #[test]
    fn display_matches_to_string() -> Result<(), Error> {
        let text = Text::from_str_with("hello", r(0), t(1))?;
        assert_eq!(format!("{text}"), text.to_string());
        Ok(())
    }

    fn arb_text() -> impl Strategy<Value = Text> {
        (0u64..3, 0usize..8).prop_map(|(replica, len)| {
            (0..len).fold(Text::empty(), |text, i| {
                let ts = i as u64;
                let ch = char::from(b'a' + ((replica.wrapping_mul(7).wrapping_add(ts)) % 26) as u8);
                text.insert(
                    i.min(text.len()),
                    ch,
                    ReplicaId::new(replica),
                    Timestamp::new(ts),
                )
                .unwrap_or(text)
            })
        })
    }

    proptest! {
        #[test]
        fn semilattice_join_is_commutative(
            a in arb_text(),
            b in arb_text(),
        ) {
            prop_assert_eq!(a.join(&b), b.join(&a));
        }

        #[test]
        fn semilattice_join_is_associative(
            a in arb_text(),
            b in arb_text(),
            c in arb_text(),
        ) {
            prop_assert_eq!(a.join(&b).join(&c), a.join(&b.join(&c)));
        }

        #[test]
        fn semilattice_join_is_idempotent(a in arb_text()) {
            prop_assert_eq!(a.join(&a), a);
        }

        #[test]
        fn semilattice_join_is_upper_bound(
            a in arb_text(),
            b in arb_text(),
        ) {
            let joined = a.join(&b);
            prop_assert!(a <= joined);
            prop_assert!(b <= joined);
        }

        #[test]
        fn insert_is_monotonic(
            text in arb_text(),
            rid in 0u64..3,
            ts in 200u64..300,
        ) {
            let ch = char::from(b'a' + ((rid.wrapping_mul(31).wrapping_add(ts) % 26) as u8));
            let idx = text.len() / 2;
            let updated = text.insert(idx, ch, ReplicaId::new(rid), Timestamp::new(ts))
                .unwrap_or(text.clone());
            prop_assert!(text <= updated);
        }

        #[test]
        fn delete_is_monotonic(text in arb_text()) {
            if !text.is_empty() {
                let updated = text.delete_at(0).unwrap_or(text.clone());
                prop_assert!(text <= updated);
            }
        }

        #[test]
        fn to_string_length_matches_len(text in arb_text()) {
            prop_assert_eq!(text.to_string().len(), text.len());
        }
    }
}
