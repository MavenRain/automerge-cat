//! Round-trip serialization tests via bincode.

use std::collections::BTreeSet;

use automerge_cat::document::{Document, Float64, NodeId, Value};
use automerge_cat::gcounter::GCounter;
use automerge_cat::lww_register::LwwRegister;
use automerge_cat::mvregister::MvRegister;
use automerge_cat::oplog::{Action, OpLog};
use automerge_cat::orset::OrSet;
use automerge_cat::pncounter::PnCounter;
use automerge_cat::replica::{ReplicaId, Tag, Timestamp};
use automerge_cat::rga::{Origin, Rga};
use automerge_cat::text::Text;

fn r(id: u64) -> ReplicaId {
    ReplicaId::new(id)
}

fn t(ts: u64) -> Timestamp {
    Timestamp::new(ts)
}

/// Serialize then deserialize via bincode.
fn round_trip<T>(val: &T) -> Result<T, String>
where
    T: serde::Serialize + serde::de::DeserializeOwned,
{
    let bytes = bincode::serialize(val).map_err(|e| e.to_string())?;
    bincode::deserialize(&bytes).map_err(|e| e.to_string())
}

/// Map any Display-able error to String.
fn s<E: core::fmt::Display>(e: E) -> String {
    e.to_string()
}

#[test]
fn gcounter_round_trips() -> Result<(), String> {
    let gc = GCounter::empty().increment(r(0)).increment(r(0)).increment(r(1));
    let gc2 = round_trip(&gc)?;
    assert_eq!(gc, gc2);
    Ok(())
}

#[test]
fn pncounter_round_trips() -> Result<(), String> {
    let pc = PnCounter::empty().increment(r(0)).decrement(r(1));
    let pc2 = round_trip(&pc)?;
    assert_eq!(pc, pc2);
    Ok(())
}

#[test]
fn lww_register_round_trips() -> Result<(), String> {
    let reg = LwwRegister::new(42_u64, t(5), r(0));
    let reg2: LwwRegister<u64> = round_trip(&reg)?;
    assert_eq!(reg, reg2);
    Ok(())
}

#[test]
fn mvregister_round_trips() -> Result<(), String> {
    let reg = MvRegister::new(1_u64, r(0), t(1)).write(2, r(1), t(2));
    let reg2: MvRegister<u64> = round_trip(&reg)?;
    assert_eq!(reg, reg2);
    Ok(())
}

#[test]
fn orset_round_trips() -> Result<(), String> {
    let set = OrSet::empty()
        .add(1_u64, r(0), t(1))
        .add(2, r(1), t(2))
        .remove(&1);
    let set2: OrSet<u64> = round_trip(&set)?;
    assert_eq!(set, set2);
    Ok(())
}

#[test]
fn rga_round_trips() -> Result<(), String> {
    let a_tag = Tag::new(r(0), t(1));
    let rga = Rga::empty()
        .insert_after(Origin::Head, 10_u64, r(0), t(1))
        .insert_after(Origin::After(a_tag), 20, r(0), t(2));
    let rga2: Rga<u64> = round_trip(&rga)?;
    assert_eq!(rga, rga2);
    Ok(())
}

#[test]
fn text_round_trips() -> Result<(), String> {
    let text = Text::from_str_with("hello world", r(0), t(1)).map_err(s)?;
    let text2 = round_trip(&text)?;
    assert_eq!(text, text2);
    assert_eq!(text2.to_string(), "hello world");
    Ok(())
}

#[test]
fn document_round_trips() -> Result<(), String> {
    let doc = Document::new();
    let (doc, list_id) = doc.create_list(r(0), t(1)).map_err(s)?;
    let doc = doc
        .set_key(NodeId::Root, "name", &Value::Str("alice".into()), r(0), t(2))
        .map_err(s)?
        .set_key(NodeId::Root, "score", &Value::Float(Float64::new(99.5)), r(0), t(3))
        .map_err(s)?
        .set_key(NodeId::Root, "items", &Value::List(list_id), r(0), t(4))
        .map_err(s)?
        .list_insert(list_id, Origin::Head, Value::Int(1), r(0), t(5))
        .map_err(s)?
        .list_insert(list_id, Origin::Head, Value::Int(2), r(0), t(6))
        .map_err(s)?;
    let doc2 = round_trip(&doc)?;
    assert_eq!(doc, doc2);
    Ok(())
}

#[test]
fn oplog_round_trips() -> Result<(), String> {
    let log = OpLog::empty()
        .append(
            Action::SetKey {
                node: NodeId::Root,
                key: "x".into(),
                value: Value::Int(42),
            },
            r(0),
            t(1),
        )
        .map_err(s)?
        .append(
            Action::SetKey {
                node: NodeId::Root,
                key: "y".into(),
                value: Value::Str("hello".into()),
            },
            r(0),
            t(2),
        )
        .map_err(s)?;
    let log2 = round_trip(&log)?;
    assert_eq!(log, log2);
    Ok(())
}

#[test]
fn merged_oplog_round_trips() -> Result<(), String> {
    use comp_cat_rs::foundation::JoinSemilattice;

    let base_heads = BTreeSet::new();
    let a = OpLog::empty()
        .append_with_deps(
            Action::SetKey {
                node: NodeId::Root,
                key: "x".into(),
                value: Value::Int(1),
            },
            r(0),
            t(1),
            base_heads.clone(),
        )
        .map_err(s)?;
    let b = OpLog::empty()
        .append_with_deps(
            Action::SetKey {
                node: NodeId::Root,
                key: "y".into(),
                value: Value::Int(2),
            },
            r(1),
            t(1),
            base_heads,
        )
        .map_err(s)?;
    let merged = a.join(&b);
    let merged2 = round_trip(&merged)?;
    assert_eq!(merged, merged2);
    assert_eq!(merged2.materialize(), merged.materialize());
    Ok(())
}
