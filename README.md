# automerge-cat

CRDTs as categorical colimits, built on [comp-cat-rs](https://crates.io/crates/comp-cat-rs).

Every data type in this crate is a join-semilattice whose merge operation is a colimit (left Kan extension) in the posetal category of states.  The categorical justification lives in `comp_cat_rs::collapse::join_is_colimit`; this crate provides the concrete data types, document model, operation log, sync protocol, and compaction.

## Quick start

```rust
use automerge_cat::{Session, NodeId, Value, Peer, ReplicaId};

// Alice creates a document
let alice = Session::new(ReplicaId::new(0))
    .set_key(NodeId::Root, "title", &Value::Str("Hello".into()))?
    .set_key(NodeId::Root, "count", &Value::Int(1))?;

// Bob creates a different document
let bob = Session::new(ReplicaId::new(1))
    .set_key(NodeId::Root, "author", &Value::Str("Bob".into()))?;

// Sync: Alice sends her changes to Bob
let peer = Peer::new();
let msg = peer.generate_message(alice.log());
let bob = bob.receive_sync(&msg);

// Bob now has Alice's keys + his own
assert!(bob.document().get_key(NodeId::Root, "title")?.contains(&Value::Str("Hello".into())));
assert!(bob.document().get_key(NodeId::Root, "author")?.contains(&Value::Str("Bob".into())));
```

## Architecture

```
comp-cat-rs (foundation)
  JoinSemilattice         trait: partial order + binary least upper bound
  collapse/join_is_colimit  join = coproduct = colimit = left Kan extension
  collapse/free_category    operations as morphisms in a free category

automerge-cat
  replica         ReplicaId, Timestamp, Tag (causal identity primitives)
  gcounter        Grow-only counter (pointwise max semilattice)
  pncounter       Positive-negative counter (product of two GCounters)
  lww_register    Last-writer-wins register (total-order semilattice)
  mvregister      Multi-value register (preserves concurrent writes)
  orset           Observed-remove set (add-wins, entries + tombstones)
  rga             Replicated growable array (sequence CRDT, tree-ordered)
  text            Collaborative text (string-level API over Rga<char>)
  document        Document model (nested maps + lists + text)
  oplog           Operation log with causal graph (CmRDT layer)
  session         Ergonomic builder (auto-clock, Document + OpLog in lockstep)
  sync            Replica-to-replica sync protocol (Peer, SyncMessage, Change)
  error           Hand-rolled Error enum (NodeNotFound, WrongNodeType, etc.)
```

## CRDT types

| Type | Semilattice structure | Use case |
|---|---|---|
| `GCounter` | Pointwise max | Distributed counters (increment only) |
| `PnCounter` | Product of two GCounters | Distributed counters (increment + decrement) |
| `LwwRegister<A>` | Total order on (timestamp, replica) | Simple scalar fields |
| `MvRegister<A>` | Grow-only entries + tombstones | Map values (preserves concurrent writes) |
| `OrSet<A>` | Grow-only entries + tombstones | Set-valued fields (add-wins) |
| `Rga<A>` | Grow-only entries + tombstones + tree | Ordered lists |
| `Text` | Delegates to `Rga<char>` | Collaborative text editing |

Every type implements `comp_cat_rs::foundation::JoinSemilattice` and all semilattice laws (commutativity, associativity, idempotency) are verified via proptest.

## Document model

The `Document` type composes `MvRegister` (for map key-value pairs) and `Rga` (for ordered lists) into a nested tree:

```rust
let s = Session::new(ReplicaId::new(0));
let (s, list_id) = s.create_list()?;
let s = s
    .set_key(NodeId::Root, "items", &Value::List(list_id))?
    .list_insert(list_id, Origin::Head, Value::Str("first".into()))?
    .list_insert(list_id, Origin::Head, Value::Str("second".into()))?;
```

Concurrent edits to the same key produce multiple values (MvRegister semantics).  A subsequent write resolves the conflict.

## Operation log and sync

The `OpLog` records every edit as a first-class operation with causal metadata.  In the free category over the causal graph, operations are edges (generating morphisms) and `materialize()` is the interpretation functor.

Sync is transport-agnostic.  Serialize a `SyncMessage` with any serde format and send it however you like:

```rust
// Alice generates changes Bob is missing
let mut alice_peer = Peer::new();
let msg = alice_peer.generate_message(alice.log());
alice_peer = alice_peer.record_sent(&msg);

// Serialize for the wire
let bytes = bincode::serialize(&msg)?;

// Bob receives and integrates
let msg: SyncMessage = bincode::deserialize(&bytes)?;
let bob = bob.receive_sync(&msg);
```

## Compaction

Tombstones accumulate as edits are made.  Once all replicas have observed a tombstone, it can be garbage-collected:

```rust
use automerge_cat::safe_compaction_tags;

let safe = safe_compaction_tags(session.log(), &[&peer_a, &peer_b]);
let session = session.compact(&safe);
```

## Serialization

All types implement `serde::Serialize` and `serde::Deserialize`.  Pick your format: bincode for compact binary, postcard for embedded, MessagePack, CBOR, or JSON (with a map-key adapter for non-string keys).

## Error handling

All user-facing APIs return `Result<T, Error>`.  The `Error` enum covers:

- `NodeNotFound` -- target node missing from document
- `WrongNodeType` -- map operation on a list node, or vice versa
- `NodeAlreadyExists` -- container creation with a colliding tag
- `DuplicateOp` -- appending an operation with an existing tag
- `IndexOutOfBounds` -- text index beyond visible length

## The categorical story

The merge of any two CRDT states is a **colimit** (specifically, a coproduct in the posetal category of states).  Every colimit is a **left Kan extension**.  This is proved in the Lean 4 formalization (`comp-cat-theory`) and documented in `comp_cat_rs::collapse::join_is_colimit`.

The operation log adds the **CmRDT** (operation-based) perspective.  The causal history forms a directed acyclic graph; the **free category** over this graph has operations as edges and composite edits as paths.  `OpLog::materialize()` is the `interpret` functor from the free category into the category of document states.

## License

MIT
