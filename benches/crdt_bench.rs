//! Benchmarks for `automerge-cat`: materialization, merge, text editing,
//! serialization, and compaction.

// Benchmark setup uses small bounded indices; platform-width casts are safe.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_lossless
)]

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use automerge_cat::document::{NodeId, Value};
use automerge_cat::oplog::{Action, OpLog};
use automerge_cat::replica::{ReplicaId, Timestamp};
use automerge_cat::session::Session;
use automerge_cat::sync::Peer;
use automerge_cat::text::Text;

fn r(id: u64) -> ReplicaId {
    ReplicaId::new(id)
}

/// Build a session with `n` sequential key-sets on the root map.
fn session_with_n_keys(n: u64) -> Session {
    (0..n).fold(Session::new(r(0)), |s, i| {
        s.set_key(NodeId::Root, &format!("k{i}"), &Value::Uint(i))
            .unwrap_or(s)
    })
}

/// Build an `OpLog` with `n` sequential set-key ops.
fn oplog_with_n_ops(n: u64) -> OpLog {
    (0..n).fold(OpLog::empty(), |log, i| {
        log.append(
            Action::SetKey {
                node: NodeId::Root,
                key: format!("k{i}"),
                value: Value::Uint(i),
            },
            r(0),
            Timestamp::new(i),
        )
        .unwrap_or(log)
    })
}

/// Build a `Text` with `n` characters (one insert at a time).
fn text_with_n_chars(n: usize) -> Text {
    (0..n).fold(Text::empty(), |text, i| {
        let ch = char::from(b'a' + (i % 26) as u8);
        text.insert(i, ch, r(0), Timestamp::new(i as u64))
            .unwrap_or(text)
    })
}

/// Build a `Text` with `n` characters (bulk constructor).
fn text_bulk(n: u64) -> Text {
    Text::from_chars((0..n).map(|i| {
        let ch = char::from(b'a' + (i % 26) as u8);
        (ch, r(0), Timestamp::new(i))
    }))
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

fn bench_materialize(c: &mut Criterion) {
    for &n in &[10_u64, 50, 100, 500] {
        let log = oplog_with_n_ops(n);
        c.bench_function(&format!("materialize_{n}_ops"), |b| {
            b.iter(|| black_box(log.materialize()));
        });
    }
}

fn bench_session_merge(c: &mut Criterion) {
    for &n in &[10_u64, 50, 100] {
        let a = session_with_n_keys(n);
        let b = (0..n).fold(Session::new(r(1)), |s, i| {
            s.set_key(NodeId::Root, &format!("j{i}"), &Value::Uint(i))
                .unwrap_or(s)
        });
        c.bench_function(&format!("merge_{n}_keys_each"), |bench| {
            bench.iter(|| black_box(a.merge(&b)));
        });
    }
}

fn bench_text_insert(c: &mut Criterion) {
    for &n in &[100_usize, 500, 1000] {
        c.bench_function(&format!("text_insert_{n}_chars"), |b| {
            b.iter(|| black_box(text_with_n_chars(n)));
        });
    }
}

fn bench_text_bulk(c: &mut Criterion) {
    for &n in &[100_u64, 500, 1000, 5000] {
        c.bench_function(&format!("text_bulk_{n}_chars"), |b| {
            b.iter(|| black_box(text_bulk(n)));
        });
    }
}

fn bench_text_to_string(c: &mut Criterion) {
    for &n in &[100_usize, 500, 1000] {
        let text = text_with_n_chars(n);
        c.bench_function(&format!("text_to_string_{n}_chars"), |b| {
            b.iter(|| black_box(text.to_string()));
        });
    }
}

fn bench_sync_generate(c: &mut Criterion) {
    for &n in &[10_u64, 50, 100] {
        let session = session_with_n_keys(n);
        let peer = Peer::new();
        c.bench_function(&format!("sync_generate_{n}_ops"), |b| {
            b.iter(|| black_box(peer.generate_message(session.log())));
        });
    }
}

fn bench_sync_receive(c: &mut Criterion) {
    for &n in &[10_u64, 50, 100] {
        let sender = session_with_n_keys(n);
        let receiver = Session::new(r(1));
        let peer = Peer::new();
        let msg = peer.generate_message(sender.log());
        c.bench_function(&format!("sync_receive_{n}_ops"), |b| {
            b.iter(|| black_box(receiver.receive_sync(&msg)));
        });
    }
}

fn bench_serialize_document(c: &mut Criterion) {
    for &n in &[10_u64, 50, 100] {
        let session = session_with_n_keys(n);
        let doc = session.document();
        c.bench_function(&format!("serialize_doc_{n}_keys"), |b| {
            b.iter(|| black_box(bincode::serialize(doc)));
        });
        let bytes = bincode::serialize(doc).unwrap_or_default();
        c.bench_function(&format!("deserialize_doc_{n}_keys"), |b| {
            b.iter(|| {
                black_box(
                    bincode::deserialize::<automerge_cat::document::Document>(&bytes),
                )
            });
        });
    }
}

fn bench_compaction(c: &mut Criterion) {
    for &n in &[10_u64, 50, 100] {
        let half = (n / 2).max(1);
        let session = (0..n).fold(Session::new(r(0)), |s, i| {
            let key = format!("k{}", i % half);
            s.set_key(NodeId::Root, &key, &Value::Uint(i))
                .unwrap_or(s)
        });
        let all_tags = session.log().tags();
        c.bench_function(&format!("compact_{n}_ops"), |b| {
            b.iter(|| black_box(session.compact(&all_tags)));
        });
    }
}

criterion_group!(
    benches,
    bench_materialize,
    bench_session_merge,
    bench_text_insert,
    bench_text_bulk,
    bench_text_to_string,
    bench_sync_generate,
    bench_sync_receive,
    bench_serialize_document,
    bench_compaction,
);
criterion_main!(benches);
