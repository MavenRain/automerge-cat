//! # automerge-cat
//!
//! CRDTs as categorical colimits, built on `comp-cat-rs`.
//!
//! Every CRDT in this crate is a join-semilattice whose merge
//! operation is a colimit (left Kan extension) in the posetal
//! category of states.  The categorical justification lives in
//! `comp_cat_rs::collapse::join_is_colimit`; this crate provides
//! the concrete data types.
//!
//! ## Architecture
//!
//! ```text
//! replica         Domain primitives: ReplicaId, Timestamp, Tag
//! gcounter        Grow-only counter (pointwise max semilattice)
//! pncounter       Positive-negative counter (product semilattice)
//! lww_register    Last-writer-wins register (total-order semilattice)
//! mvregister      Multi-value register (preserves concurrent writes)
//! orset           Observed-remove set (add-wins, product semilattice)
//! rga             Replicated growable array (sequence CRDT)
//! document        Document model composing maps + lists
//! text            Collaborative text (string-level API over Rga)
//! oplog           Operation log with causal graph (CmRDT layer)
//! ```
//!
//! Each type implements `comp_cat_rs::foundation::JoinSemilattice`.

pub mod error;
pub mod replica;
pub mod gcounter;
pub mod pncounter;
pub mod lww_register;
pub mod mvregister;
pub mod orset;
pub mod rga;
pub mod text;
pub mod document;
pub mod oplog;
pub mod session;
pub mod sync;

pub use error::Error;
pub use replica::{ReplicaId, Tag, Timestamp};
pub use gcounter::GCounter;
pub use pncounter::PnCounter;
pub use lww_register::LwwRegister;
pub use mvregister::MvRegister;
pub use orset::OrSet;
pub use rga::{Origin, Rga};
pub use text::Text;
pub use document::{Document, Float64, Node, NodeId, Value};
pub use oplog::{Action, Op, OpLog};
pub use session::Session;
pub use sync::{safe_compaction_tags, Change, Peer, SyncMessage};
