//! Phase timing of the compile pipeline (docs/compile-time.md).
//!
//! A per-thread, opt-in recorder: `enable(true)`, run the compile, `take()` the entries (aggregated by phase name in first-seen order: wall time and the number of runs - a retry ladder shows as
//! a run count above one). Disabled (the default) a [`time`] call costs one thread-local flag read. There is no environment switch: the caller (`amdc emit --time`, `amdc compile-batch`) enables it.

use std::cell::{Cell, RefCell};
use std::time::{Duration, Instant};

/// One phase of the pipeline: its total wall time over all its runs on this thread.
#[derive(Clone, Debug, PartialEq)]
pub struct Entry {
    pub name: &'static str,
    pub dur: Duration,
    /// how many times the phase ran (the outer / inner retry ladders run the passes again)
    pub runs: u32,
}

thread_local! {
    static ON: Cell<bool> = const { Cell::new(false) };
    static REC: RefCell<Vec<Entry>> = const { RefCell::new(Vec::new()) };
}

/// Turns recording on or off for the calling thread (clears nothing).
pub fn enable(on: bool) {
    ON.with(|c| c.set(on));
}

pub fn is_on() -> bool {
    ON.with(|c| c.get())
}

/// Records `d` under `name` (no-op when recording is off).
pub fn add(name: &'static str, d: Duration) {
    if !is_on() {
        return;
    }
    REC.with(|r| {
        let mut r = r.borrow_mut();
        match r.iter_mut().find(|e| e.name == name) {
            Some(e) => {
                e.dur += d;
                e.runs += 1;
            }
            None => r.push(Entry { name, dur: d, runs: 1 }),
        }
    });
}

/// Runs `f`, recording its wall time under `name` when recording is on.
#[inline]
pub fn time<R>(name: &'static str, f: impl FnOnce() -> R) -> R {
    if !is_on() {
        return f();
    }
    let t = Instant::now();
    let r = f();
    add(name, t.elapsed());
    r
}

/// The recorded entries of this thread (first-seen order); clears them.
pub fn take() -> Vec<Entry> {
    REC.with(|r| std::mem::take(&mut *r.borrow_mut()))
}

/// Adds `src` into `dst` (same names merge; new names append): the totals of a batch over its worker threads.
pub fn merge(dst: &mut Vec<Entry>, src: &[Entry]) {
    for e in src {
        match dst.iter_mut().find(|d| d.name == e.name) {
            Some(d) => {
                d.dur += e.dur;
                d.runs += e.runs;
            }
            None => dst.push(e.clone()),
        }
    }
}

/// The entries as an aligned table (ms, runs, share of `total`), slowest first.
pub fn table(entries: &[Entry], total: Duration) -> String {
    let mut v: Vec<&Entry> = entries.iter().collect();
    v.sort_by(|a, b| b.dur.cmp(&a.dur));
    let tot = total.as_secs_f64().max(1e-12);
    let mut s = String::new();
    s.push_str(&format!("{:<34} {:>10} {:>6} {:>6}\n", "phase", "ms", "runs", "%"));
    for e in v {
        let ms = e.dur.as_secs_f64() * 1e3;
        s.push_str(&format!("{:<34} {:>10.3} {:>6} {:>5.1}%\n", e.name, ms, e.runs, 100.0 * e.dur.as_secs_f64() / tot));
    }
    s.push_str(&format!("{:<34} {:>10.3}\n", "total (wall)", total.as_secs_f64() * 1e3));
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_records_nothing_and_enabled_aggregates() {
        enable(false);
        take();
        time("x", || ());
        assert!(take().is_empty());
        enable(true);
        time("a", || ());
        time("b", || ());
        time("a", || ());
        enable(false);
        let e = take();
        assert_eq!(e.iter().map(|x| (x.name, x.runs)).collect::<Vec<_>>(), vec![("a", 2), ("b", 1)]);
        let mut tot = Vec::new();
        merge(&mut tot, &e);
        merge(&mut tot, &e);
        assert_eq!(tot[0].runs, 4);
        assert!(table(&tot, Duration::from_millis(1)).contains("total (wall)"));
    }
}
