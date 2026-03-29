//! Project-wide error type.
//!
//! Surfaces conditions that the CRDT primitives silently absorb
//! (duplicate tags, idempotent operations) but that higher-level
//! APIs (`Document`, `OpLog`, `Text`) should report to callers.

use crate::document::NodeId;
use crate::replica::Tag;

/// Errors arising from document, oplog, or text operations.
#[derive(Debug)]
pub enum Error {
    /// The target node does not exist in the document.
    NodeNotFound {
        /// The missing node.
        node: NodeId,
    },
    /// A map operation was attempted on a list node, or vice versa.
    WrongNodeType {
        /// The node that was targeted.
        node: NodeId,
        /// What was expected (e.g. "map" or "list").
        expected: &'static str,
    },
    /// A container with this ID already exists in the document.
    NodeAlreadyExists {
        /// The colliding node.
        node: NodeId,
    },
    /// An operation with this tag already exists in the oplog.
    DuplicateOp {
        /// The colliding tag.
        tag: Tag,
    },
    /// The character index is out of bounds.
    IndexOutOfBounds {
        /// The invalid index.
        index: usize,
        /// The length of the text.
        len: usize,
    },
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NodeNotFound { node } => {
                write!(f, "node not found: {node:?}")
            }
            Self::WrongNodeType { node, expected } => {
                write!(f, "wrong node type for {node:?}: expected {expected}")
            }
            Self::NodeAlreadyExists { node } => {
                write!(f, "node already exists: {node:?}")
            }
            Self::DuplicateOp { tag } => {
                write!(f, "duplicate operation: {tag}")
            }
            Self::IndexOutOfBounds { index, len } => {
                write!(f, "index {index} out of bounds (len {len})")
            }
        }
    }
}

impl std::error::Error for Error {}
