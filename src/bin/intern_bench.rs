//===- intern_bench.rs - Vx Compiler ---------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//              EVAL-ONLY. `parallel-frontend-eval` BRANCH. NEVER MERGED.
//
// E1 + E2 of the CGO 2027 R2 measurement plan (#295, #296): time the frontend under both interning
// strategies across a corpus sweep and a thread ladder, so the deferred design's speedup has a
// baseline to attribute against and a shape to attribute it to.
//
//   cargo run --release --bin intern_bench -- \
//       --modules 8,64,512 --fns 16,128 --density 0,1 --threads 1,2,4,8 --reps 10
//
// Every swept dimension appears in the CSV, so a raw log is self-describing.
//
//===----------------------------------------------------------------------===//

//! Times the Vx frontend under both interning strategies across a corpus sweep and a thread
//! ladder, and prints a CSV of the results plus per-phase scaling tables.

#[path = "corpus/mod.rs"]
mod corpus;

use std::collections::HashMap;
use std::time::{Duration, Instant};
use vxc::config::Schedule;
use vxc::intern_mode::{self, InternMode};

/// One cell of the sweep: the interning mode and the thread count, with `0` for the
/// sequential arm.
type CellKey = (&'static str, usize);

/// The median time of each phase in one cell, in milliseconds, in pipeline order.
type PhaseMedians = Vec<(&'static str, f64)>;

/// The time each phase took in one rep, as the compiler recorded it.
type PhaseTimings = Vec<(&'static str, Duration)>;

/// The median and the first and third quartiles of `xs`, in that order.
///
/// # Panics
///
/// Panics if a sample is NaN, which has no place in a sort order.
fn median_iqr(mut xs: Vec<f64>) -> (f64, f64, f64) {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q = |p: f64| -> f64 {
        if xs.is_empty() {
            return 0.0;
        }
        let i = quantile_index(p, xs.len());
        xs[i.min(xs.len() - 1)]
    };
    (q(0.5), q(0.25), q(0.75))
}

/// The index of the `p` quantile (nearest rank) in a sorted list of `len` samples.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "a sample count is far below 2^52, and p is between 0 and 1, so the rounded product is a whole number between 0 and len - 1"
)]
fn quantile_index(p: f64, len: usize) -> usize {
    (p * (len - 1) as f64).round() as usize
}

/// Phase medians for one cell, in the order the pipeline runs them. A phase no rep recorded is
/// left out: the table names the driver's backend phases too, which only `vxc` runs.
fn phase_report(samples: &[PhaseTimings]) -> PhaseMedians {
    let mut names: Vec<&'static str> = Vec::new();
    for s in samples {
        for (n, _) in s {
            if !names.contains(n) {
                names.push(n);
            }
        }
    }
    names
        .into_iter()
        .filter_map(|n| {
            let xs: Vec<f64> = samples
                .iter()
                .map(|s| {
                    s.iter()
                        .filter(|(m, _)| *m == n)
                        .map(|(_, d)| d.as_secs_f64() * 1e3)
                        .sum::<f64>()
                })
                .collect();
            xs.iter().any(|x| *x > 0.0).then(|| (n, median_iqr(xs).0))
        })
        .collect()
}

/// The phases to give a row in the scaling table: every phase any cell of the ladder recorded, in
/// the order the pipeline runs them.
///
/// Taking the rows from the baseline cell alone dropped a phase that measured zero there from
/// every column, including the thread counts where it ran and was not zero. Its time still joined
/// those columns' totals, so the phase was counted everywhere and named nowhere.
fn scaling_rows(cells: &[&Vec<(&'static str, f64)>]) -> Vec<&'static str> {
    let mut rows: Vec<&'static str> = Vec::new();
    for cell in cells {
        for (name, _) in *cell {
            if !rows.contains(name) {
                rows.push(name);
            }
        }
    }
    rows.sort_by_key(|n| phase_order(n));
    rows
}

/// Where a phase sits in the pipeline. An unknown name sorts last rather than panicking: this is
/// a report, and a name the compiler timed but did not declare is the phase table's problem to
/// report, not this function's to crash on.
fn phase_order(name: &str) -> usize {
    vxc::intern_mode::PHASES
        .iter()
        .position(|p| *p == name)
        .unwrap_or(usize::MAX)
}

/// What one rep compiled: its wall clock, and how many bytes of MLIR came out.
///
/// `Some(0)` means codegen ran and the flat emitter *declined* — the frontend still did all its
/// work, but nothing was generated, so the rep timed an incomplete compile and must not be quoted
/// as one. `None` means codegen was not asked for (`--emit=none`).
struct Rep {
    wall: Duration,
    mlir_bytes: Option<usize>,
}

/// One compile, timed, inside a caller-supplied pool.
///
/// The pool is the caller's because thread *placement* is the dominant nuisance variable on a
/// heterogeneous machine, and it is decided when the pool is built. On an Apple M4 (4 performance +
/// 6 efficiency cores) a two-thread pool lands either on two P-cores or not, and the difference is a
/// factor of 1.7 on this corpus -- measured: the same binary, corpus and mode gives a median of
/// 30 ms in one process and 50 ms in the next, with under 2% spread *within* each. A per-rep pool
/// makes that draw a hidden per-rep variable; a per-cell pool makes it a constant that every mode in
/// the cell shares, so a mode comparison is paired against it rather than confounded by it.
///
/// This is not hypothetical caution. Running each mode's whole ladder in sequence produced a table
/// showing deferred gaining nothing from a second core while content gained 1.6x, reproducibly
/// across reps and across three runs. Both modes show the same bimodality when measured alone: the
/// effect was the placement draw, not the design.
///
/// # Panics
///
/// Panics if the compile fails. The corpus is generated to compile, so a failure is a compiler
/// bug, not a measurement.
fn run_in(
    pool: Option<&rayon::ThreadPool>,
    paths: &[String],
    emit_mlir: bool,
    sched: Schedule,
    mode: InternMode,
) -> Rep {
    let _ = intern_mode::take_phases(); // drop any timings from a prior rep
    let t = Instant::now();
    let compile = || {
        if emit_mlir {
            Some(
                vxc::pipeline::compile_pipeline_mlir_in(paths, sched, mode)
                    .expect("pipeline")
                    .map_or(0, |t| t.len()),
            )
        } else {
            let _ = vxc::pipeline::compile_pipeline_type_stream_in(paths, sched, mode)
                .expect("pipeline");
            None
        }
    };
    // The sequential arm runs *outside* any pool. Installing into a rayon pool that is then never
    // asked to do parallel work would still be a rayon run, and the point of the arm is to have
    // rayon nowhere on the path -- otherwise its cost is charged to both sides of the comparison and
    // cancels out of every ratio.
    let mlir_bytes = match pool {
        Some(p) if sched != Schedule::Sequential => p.install(compile),
        _ => compile(),
    };
    Rep {
        wall: t.elapsed(),
        mlir_bytes,
    }
}

/// The machine's own scaling ceiling: a parallel-for that allocates nothing, touches no shared
/// state, and fits its whole working set in registers.
///
/// Whatever this reaches at N threads is the most *any* phase can reach, so a phase's scaling should
/// be read as a fraction of it rather than as a fraction of N. Two effects are charged here where
/// they are visible instead of silently inflating the compiler's apparent serial fraction: clock
/// scaling (one active core boosts higher than four) and shared cache/bandwidth. Without this
/// control, "4 threads gave 2.6x" reads as 35% of the compiler failing to parallelise, when much of
/// it may be a ceiling no program on this machine can beat.
///
/// The work is a splitmix64 chain, which is a dependent sequence of integer ops -- no memory
/// traffic, no branch prediction to win, and impossible to vectorise away. `black_box` on the result
/// keeps it from being optimised out entirely.
fn calibrate(pool: Option<&rayon::ThreadPool>, items: usize, iters: u64) -> Duration {
    use rayon::prelude::*;
    let burn = |seed: u64| -> u64 {
        let mut x = seed;
        for _ in 0..iters {
            x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = x;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            x = x.wrapping_add(z ^ (z >> 31));
        }
        x
    };
    let t = Instant::now();
    // Wrapping, like every step of `burn` itself: this is a calibration workload whose only job
    // is to take time, and its total is fed straight to `black_box`. A plain `sum()` overflows
    // `u64` and panics in a debug build, which took the whole harness down before it measured
    // anything.
    let total: u64 = match pool {
        Some(p) => p.install(|| {
            (0..items)
                .into_par_iter()
                .map(|i| burn(i as u64))
                .reduce(|| 0u64, u64::wrapping_add)
        }),
        None => (0..items)
            .map(|i| burn(i as u64))
            .fold(0u64, u64::wrapping_add),
    };
    std::hint::black_box(total);
    t.elapsed()
}

/// Everything the command line chose, read once before any measurement.
struct Options {
    /// The raw command line, for the corpus parameters that `corpus` reads itself.
    args: Vec<String>,
    module_counts: Vec<usize>,
    fn_counts: Vec<usize>,
    densities: Vec<f64>,
    reps: usize,
    threads: Vec<usize>,
    emit_mlir: bool,
    run_seq: bool,
    run_par: bool,
    modes: Vec<(&'static str, InternMode)>,
    /// The thread counts to measure, with `0` standing for the sequential arm.
    ladder: Vec<usize>,
    cores: usize,
}

/// Reads the sweep from the command line.
///
/// # Panics
///
/// Panics when `--modes` names no mode, or when the schedule leaves no thread count to run.
fn parse_options() -> Options {
    let args: Vec<String> = std::env::args().collect();
    let module_counts: Vec<usize> = corpus::arg_list(&args, "--modules", vec![64]);
    let fn_counts: Vec<usize> = corpus::arg_list(&args, "--fns", vec![16]);
    // The generics-density sweep is the comparison E2 exists for: the same function count with and
    // without instantiations in the signatures.
    let densities: Vec<f64> = corpus::arg_list(&args, "--density", vec![1.0]);
    let reps: usize = corpus::arg(&args, "--reps", 10);
    let threads: Vec<usize> = corpus::arg_list(&args, "--threads", vec![1, 2, 4, 8]);
    // Codegen is on by default (#311). A sweep that stops at the SIMD patch times a frontend, not a
    // compile, and the number it produces cannot be quoted as a compile-time speedup no matter how
    // carefully the rest of the harness is built. `--emit=none` reproduces the old frontend-only
    // measurement when the frontend is deliberately what is under study.
    let emit_mlir = corpus::arg::<String>(&args, "--emit", "mlir".to_string()) != "none";
    // `both` (default) runs a rayon-free arm before the thread ladder, so the ladder's baseline is
    // "the same compiler, not parallelised" rather than "the same compiler, parallelised, on one
    // thread". Without it the parallel machinery's own cost sits on both sides of every ratio and
    // cancels, and the sweep can only answer whether more threads help -- never whether any of this
    // beats not doing it. `par` / `seq` run one arm alone.
    let schedule_arg = corpus::arg::<String>(&args, "--schedule", "both".to_string());
    let run_seq = schedule_arg != "par";
    let run_par = schedule_arg != "seq";
    // Which interning modes to run, and in which order. Both in one process is convenient and
    // produces the comparison table, but it is also a confound: the second mode's cells run on a
    // heap the first mode fragmented and a machine the first mode warmed. `--modes content` /
    // `--modes deferred` puts each in its own process, and `--modes content,deferred` reverses the
    // order -- between them, whether an effect follows the *mode* or its *position in the run* is
    // decidable rather than assumed.
    let modes: Vec<(&'static str, InternMode)> =
        corpus::arg::<String>(&args, "--modes", "deferred,content".to_string())
            .split(',')
            .filter_map(|m| match m.trim() {
                "deferred" => Some(("deferred", InternMode::Deferred)),
                "content" => Some(("content", InternMode::Content)),
                other => {
                    eprintln!("unknown --modes entry '{other}', expected deferred or content");
                    None
                }
            })
            .collect();
    assert!(!modes.is_empty(), "--modes selected nothing to run");

    // The ladder, with `0` standing for the sequential arm: one key space for the medians map, and
    // `0 threads` reads as "no thread pool", which is what it is.
    let mut ladder: Vec<usize> = Vec::new();
    if run_seq {
        ladder.push(0);
    }
    if run_par {
        ladder.extend(threads.iter().copied());
    }
    assert!(
        !ladder.is_empty(),
        "--schedule and --threads selected no thread count to run"
    );

    let cores = std::thread::available_parallelism().map_or(1, std::num::NonZero::get);
    Options {
        args,
        module_counts,
        fn_counts,
        densities,
        reps,
        threads,
        emit_mlir,
        run_seq,
        run_par,
        modes,
        ladder,
        cores,
    }
}

/// Warns when a thread count exceeds the machine's cores: past the core count a row measures
/// oversubscription, not scaling.
fn warn_if_oversubscribed(opts: &Options) {
    let over = opts.threads.iter().filter(|&&t| t > opts.cores).count();
    if over > 0 {
        eprintln!(
            "WARNING: {over} of {} thread counts exceed this machine's {} cores. Past the \
             core count you are measuring oversubscription, not scaling — those rows are not \
             publishable data. Run the full ladder on a machine with at least as many cores as \
             the largest thread count.",
            opts.threads.len(),
            opts.cores
        );
    }
}

/// A rayon pool of `t` threads, or none for the sequential arm (`t == 0`).
///
/// # Panics
///
/// Panics if the operating system refuses to start the threads.
fn build_pool(t: usize) -> Option<rayon::ThreadPool> {
    (t > 0).then(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(t)
            .build()
            .expect("thread pool")
    })
}

/// Measures the machine's own scaling ceiling on every step of the ladder and prints it, so
/// every scaling number below can be read against what this machine can actually deliver rather
/// than against N; see `calibrate`.
///
/// # Panics
///
/// Panics if a thread pool cannot be built, or if a timing is NaN, which a clock never gives.
fn print_machine_ceiling(ladder: &[usize]) {
    let items = 1600; // one work item per function in the default corpus
    let mut cal: Vec<(usize, f64)> = Vec::new();
    for &t in ladder {
        let pool = build_pool(t);
        let _ = calibrate(pool.as_ref(), items, 4_000); // warm up
        let mut xs: Vec<f64> = (0..5)
            .map(|_| calibrate(pool.as_ref(), items, 4_000).as_secs_f64() * 1e3)
            .collect();
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        cal.push((t, xs[xs.len() / 2]));
    }
    let base = cal[0].1;
    eprint!("machine ceiling (pure compute, no allocation):");
    for (t, ms) in &cal {
        eprint!(
            "  {}={:.2}x",
            if *t == 0 {
                "seq".to_string()
            } else {
                format!("t{t}")
            },
            base / ms
        );
    }
    eprintln!(
        "\n  Read every phase speedup below as a fraction of this, not of the thread \
               count: clock scaling and shared cache are charged here, where they belong."
    );
}

/// One generated corpus and the swept dimensions that produced it, as the CSV names them.
struct CorpusCell<'a> {
    modules: usize,
    fns: usize,
    density: f64,
    corpus: &'a corpus::Corpus,
}

/// What one corpus measured: the median wall clock and the per-phase medians of every cell.
#[derive(Default)]
struct Results {
    medians: HashMap<CellKey, f64>,
    // Per-phase medians, so each phase gets its own scaling curve. The total's speedup
    // says only *that* something is not parallelising; this says *which*, and a phase
    // whose own curve is flat is a different problem from one that scales but is small.
    phase_medians: HashMap<CellKey, PhaseMedians>,
}

impl Results {
    /// The median wall clock of one cell in milliseconds, or NaN when it was not run.
    fn at(&self, mode: &'static str, t: usize) -> f64 {
        self.medians.get(&(mode, t)).copied().unwrap_or(f64::NAN)
    }
}

/// Every rep of one thread count, indexed by mode: the wall clocks in milliseconds, the phase
/// timings, and the size of the MLIR the last rep produced.
struct RepSamples {
    wall_ms: Vec<Vec<f64>>,
    phases: Vec<Vec<PhaseTimings>>,
    mlir_bytes: Vec<usize>,
}

/// Every mode, `reps` times each, alternating rep by rep in the given pool. One warm-up rep per
/// mode runs first and is discarded: the first run pays page faults and filesystem-cache misses
/// that have nothing to do with interning.
fn run_reps(
    opts: &Options,
    c: &corpus::Corpus,
    pool: Option<&rayon::ThreadPool>,
    sched: Schedule,
) -> RepSamples {
    let mut samples: Vec<Vec<f64>> = vec![Vec::with_capacity(opts.reps); opts.modes.len()];
    let mut phase_samples: Vec<Vec<_>> = vec![Vec::with_capacity(opts.reps); opts.modes.len()];
    let mut mlir_bytes = vec![0usize; opts.modes.len()];

    for &(_, mode) in &opts.modes {
        let warm = run_in(pool, &c.paths, opts.emit_mlir, sched, mode);
        if warm.mlir_bytes == Some(0) {
            eprintln!(
                "  NOTE: the flat emitter declined this corpus, so every rep below \
                 times a frontend plus a codegen attempt that produced nothing. Not \
                 a compile-time measurement. Run with VX_FLAT_DBG=1 to see which \
                 construct declined, or --emit=none to measure the frontend on \
                 purpose."
            );
        }
    }
    for _ in 0..opts.reps {
        for (i, &(_, mode)) in opts.modes.iter().enumerate() {
            let rep = run_in(pool, &c.paths, opts.emit_mlir, sched, mode);
            mlir_bytes[i] = rep.mlir_bytes.unwrap_or(0);
            samples[i].push(rep.wall.as_secs_f64() * 1e3);
            phase_samples[i].push(intern_mode::take_phases());
        }
    }
    RepSamples {
        wall_ms: samples,
        phases: phase_samples,
        mlir_bytes,
    }
}

/// Prints one CSV row and one phase line per mode for one thread count, and records the medians
/// for the scaling tables.
fn report_thread_count(
    opts: &Options,
    cell: &CorpusCell<'_>,
    t: usize,
    runs: &RepSamples,
    results: &mut Results,
) {
    let tag = if t == 0 {
        "seq".to_string()
    } else {
        t.to_string()
    };
    for (i, &(label, _)) in opts.modes.iter().enumerate() {
        let mlir_bytes = runs.mlir_bytes[i];
        let (med, q1, q3) = median_iqr(runs.wall_ms[i].clone());
        results.medians.insert((label, t), med);
        println!(
            "{label},{},{},{},{},{},{tag},{med:.2},{q1:.2},{q3:.2},{}",
            cell.modules,
            cell.fns,
            cell.density,
            cell.corpus.generic_slots,
            cell.corpus.distinct_keys,
            opts.reps
        );
        let ph = phase_report(&runs.phases[i]);
        results.phase_medians.insert((label, t), ph.clone());
        let total: f64 = ph.iter().map(|(_, v)| v).sum();
        let share = |name: &str| -> f64 {
            let v: f64 = ph.iter().filter(|(n, _)| *n == name).map(|(_, v)| v).sum();
            if total > 0.0 {
                v / total * 100.0
            } else {
                0.0
            }
        };
        eprintln!(
            "  [{label} t={tag}] {} | measured-phase total {:.1} ms, type_check \
             {:.0}%, codegen {:.0}% (the rest is serial or barrier work){}",
            ph.iter()
                .map(|(n, v)| format!("{n} {v:.1}"))
                .collect::<Vec<_>>()
                .join("  "),
            total,
            share("type_check"),
            share("codegen"),
            if opts.emit_mlir {
                format!(", {mlir_bytes} bytes of MLIR")
            } else {
                String::new()
            }
        );
    }
}

/// One thread count on one corpus. One pool per cell, and every mode in the cell runs in that
/// pool, alternating rep by rep -- so whatever placement the pool drew, both modes drew it, and
/// the mode comparison is paired rather than confounded by a nuisance variable worth 1.7x on
/// this machine. Running each mode's ladder end to end instead once produced a clean,
/// reproducible, and entirely false finding; see `run_in`.
fn measure_thread_count(opts: &Options, cell: &CorpusCell<'_>, t: usize, results: &mut Results) {
    let sched = if t == 0 {
        Schedule::Sequential
    } else {
        Schedule::Parallel
    };
    let pool = build_pool(t);
    let runs = run_reps(opts, cell.corpus, pool.as_ref(), sched);
    report_thread_count(opts, cell, t, &runs, results);
}

/// Names the baseline column: the sequential arm when it ran, else the first thread count.
fn baseline_tag(opts: &Options) -> &'static str {
    if opts.ladder[0] == 0 {
        "sequential, no rayon"
    } else {
        "1 thread (rayon)"
    }
}

/// The attribution number. Speedup is measured against each mode's own *sequential* run when
/// there is one -- the same compiler with rayon off the path -- so the ratio answers "is
/// parallelising this worth it", not merely "does adding threads help a design that is already
/// paying for parallelism". Falling back to the first thread count under `--schedule par` keeps
/// that arm readable, but its ratios are the weaker claim and the header says which is in force.
///
/// Returns the thread counts whose row is suspect: slower than a row with fewer threads.
fn print_scaling_table(opts: &Options, results: &Results) -> Vec<usize> {
    let base_key = opts.ladder[0];
    let base_tag = baseline_tag(opts);
    eprintln!(
        "  threads  sp(deferred)  sp(content)   ms(deferred/content)   [baseline: {base_tag}]"
    );
    let base = |mm: &'static str| results.at(mm, base_key);
    let mut suspect = Vec::new();
    for &t in &opts.ladder {
        // A row slower than a row with *fewer* threads is not a result. More workers
        // cannot make the same work take longer, so the row is measuring something other
        // than the compiler -- on a heterogeneous machine, almost always which cores the
        // pool drew. Flagged rather than silently tabulated, because such a row is
        // internally consistent (tight IQR, reproducible across reps) and reads exactly
        // like a finding.
        let regressed = opts.modes.iter().any(|&(mm, _)| {
            opts.ladder
                .iter()
                .take_while(|&&u| u < t)
                .any(|&u| results.at(mm, t) > results.at(mm, u) * 1.02)
        });
        if regressed {
            suspect.push(t);
        }
        let flag = if t > opts.cores {
            "  <- oversubscribed"
        } else if regressed {
            "  <- SUSPECT: slower than a lower thread count"
        } else {
            ""
        };
        let tag = if t == 0 { "seq".into() } else { t.to_string() };
        eprintln!(
            "  {tag:>7}  {:>12.2}  {:>11.2}   {:>8.1}/{:>8.1}{flag}",
            base("deferred") / results.at("deferred", t),
            base("content") / results.at("content", t),
            results.at("deferred", t),
            results.at("content", t),
        );
    }
    suspect
}

/// What the scaling table is to be read against: the parallel machinery's own overhead, the
/// absolute baselines, and a warning for every suspect row.
fn print_scaling_notes(opts: &Options, results: &Results, suspect: &[usize]) {
    let base = |mm: &'static str| results.at(mm, opts.ladder[0]);
    if opts.run_seq && opts.run_par {
        // What one rayon thread costs over no rayon at all. If it is near 1.00 the two
        // baselines are interchangeable and the scaling column can be read as speedup;
        // if it is not, every ratio taken against the 1-thread column was quietly
        // discounting the parallel machinery's own overhead.
        let overhead = |mm: &'static str| results.at(mm, opts.threads[0]) / results.at(mm, 0);
        eprintln!(
            "  parallel-machinery overhead at {} thread(s) vs no rayon: deferred {:.3}x, \
             content {:.3}x (1.00 = the parallel structure is free when not used)",
            opts.threads[0],
            overhead("deferred"),
            overhead("content"),
        );
    }
    eprintln!(
        "  baseline: deferred {:.1} ms, content {:.1} ms. Absolute times matter as much \
         as the speedup ratios here: `content` removes work rather than parallelising \
         it, so a win shows up first in the baseline column, not in the scaling column.",
        base("deferred"),
        base("content"),
    );
    if !suspect.is_empty() {
        eprintln!(
            "  WARNING: thread count(s) {suspect:?} came out slower than a lower one. \
             Adding workers cannot lengthen the same work, so such a row is measuring \
             something other than the compiler. Two causes seen so far: (a) a \
             heterogeneous host, where a small pool that draws efficiency cores runs \
             ~1.7x slower for the whole life of the pool with a tight per-rep spread \
             that reads exactly like a result; (b) too little work per thread, where the \
             corpus is small enough that pool dispatch dominates -- check the corpus \
             line above for functions-per-thread before blaming the machine. Re-run the \
             cell. Do not quote it."
        );
    }
}

/// The time of the top-level phases in one cell. Sub-phases are indented and nested *inside*
/// their parent, so they must not join the total or the shares exceed 100% and `unaccounted`
/// goes negative.
fn top_level_total(p: &PhaseMedians) -> f64 {
    p.iter()
        .filter(|(n, _)| !n.starts_with(' '))
        .map(|(_, v)| v)
        .sum()
}

/// The median of one phase in one cell, or NaN when the cell or the phase was not measured.
fn phase_at(results: &Results, label: &'static str, t: usize, name: &str) -> f64 {
    results
        .phase_medians
        .get(&(label, t))
        .and_then(|p| p.iter().find(|(n, _)| *n == name))
        .map_or(f64::NAN, |(_, v)| *v)
}

/// What every row of one per-phase table is measured against.
struct PhaseBaseline<'a> {
    label: &'static str,
    phases: &'a PhaseMedians,
    /// The top-level total of the baseline cell.
    total: f64,
    /// The largest thread count, whose column the `sp(max)` ratio uses.
    max_t: usize,
}

/// The column headings of a per-phase table: one per thread count, then the speedup at the
/// largest and the share of the baseline.
fn print_phase_header(ladder: &[usize]) {
    eprint!("  {:<16}", "phase");
    for &t in ladder {
        eprint!(
            "{:>9}",
            if t == 0 {
                "seq".to_string()
            } else {
                format!("t{t}")
            }
        );
    }
    eprintln!("{:>9}{:>10}", "sp(max)", "share");
}

/// One row of a per-phase table: the phase's median at every thread count, its speedup at the
/// largest, and its share of the baseline.
fn print_phase_row(opts: &Options, results: &Results, base: &PhaseBaseline<'_>, name: &str) {
    let base_ms = base
        .phases
        .iter()
        .find(|(n, _)| *n == name)
        .map_or(0.0, |(_, v)| *v);
    eprint!("  {name:<16}");
    for &t in &opts.ladder {
        let v = phase_at(results, base.label, t, name);
        eprint!("{v:>9.1}");
    }
    let at_max = phase_at(results, base.label, base.max_t, name);
    eprintln!(
        "{:>9.2}{:>9.1}%",
        base_ms / at_max,
        if base.total > 0.0 {
            base_ms / base.total * 100.0
        } else {
            0.0
        }
    );
}

/// The last row of a per-phase table: the wall clock the phase timers do not name. It is not a
/// rounding artifact to be ignored: work outside every `timed()` region is invisible to the
/// attribution, so a large unaccounted share means the phase table is answering a narrower
/// question than it appears to.
fn print_unaccounted_row(opts: &Options, results: &Results, base: &PhaseBaseline<'_>) {
    let label = base.label;
    let base_key = opts.ladder[0];
    eprint!("  {:<16}", "unaccounted");
    for &t in &opts.ladder {
        let named = results
            .phase_medians
            .get(&(label, t))
            .map_or(f64::NAN, top_level_total);
        eprint!("{:>9.1}", results.at(label, t) - named);
    }
    let named_base: f64 = base.total;
    let named_max: f64 = results
        .phase_medians
        .get(&(label, base.max_t))
        .map_or(f64::NAN, top_level_total);
    eprintln!(
        "{:>9.2}{:>9.1}%",
        (results.at(label, base_key) - named_base) / (results.at(label, base.max_t) - named_max),
        (results.at(label, base_key) - named_base) / results.at(label, base_key) * 100.0
    );
}

/// Per-phase scaling for one mode. The total's speedup says only *that* something is not
/// parallelising; this says which. A phase that scales but is tiny and a phase that is large and
/// flat both show up as "the total fell short", and they are entirely different problems -- the
/// first is nothing to do, the second is the whole job.
///
/// # Panics
///
/// Panics if the ladder is empty, which `parse_options` rules out.
fn print_phase_scaling(opts: &Options, results: &Results, label: &'static str) {
    let base_key = opts.ladder[0];
    let Some(base_ph) = results.phase_medians.get(&(label, base_key)) else {
        return;
    };
    eprintln!(
        "\n  per-phase scaling [{label}], baseline = {}:",
        baseline_tag(opts)
    );
    print_phase_header(&opts.ladder);
    let base = PhaseBaseline {
        label,
        phases: base_ph,
        total: top_level_total(base_ph),
        max_t: *opts.ladder.last().expect("the ladder is never empty"),
    };
    let cells: Vec<&PhaseMedians> = std::iter::once(base_key)
        .chain(opts.ladder.iter().copied())
        .filter_map(|cell| results.phase_medians.get(&(label, cell)))
        .collect();
    let rows = scaling_rows(&cells);
    for name in &rows {
        print_phase_row(opts, results, &base, name);
    }
    print_unaccounted_row(opts, results, &base);
}

fn main() {
    let opts = parse_options();
    warn_if_oversubscribed(&opts);

    // Calibrate first, on the same ladder, so every scaling number below can be read against what
    // this machine can actually deliver rather than against N.
    print_machine_ceiling(&opts.ladder);

    println!(
        "mode,modules,fns,density,generic_slots,distinct_keys,threads,median_ms,q1_ms,q3_ms,reps"
    );

    for &n in &opts.module_counts {
        for &m in &opts.fn_counts {
            for &d in &opts.densities {
                let params = corpus::params_from_args(&opts.args, n, m);
                let params = corpus::CorpusParams {
                    density: d,
                    ..params
                };
                let c = corpus::generate(&params, None);
                eprintln!("\n{}", c.log_line());
                let cell = CorpusCell {
                    modules: n,
                    fns: m,
                    density: d,
                    corpus: &c,
                };

                // Thread count outermost, modes paired inside it; see `measure_thread_count`.
                let mut results = Results::default();
                for &t in &opts.ladder {
                    measure_thread_count(&opts, &cell, t, &mut results);
                }

                let suspect = print_scaling_table(&opts, &results);
                print_scaling_notes(&opts, &results, &suspect);
                for &(label, _) in &opts.modes {
                    print_phase_scaling(&opts, &results, label);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A cell leaves out a phase it measured as zero, so the baseline cell is not the full set of
    /// phases the ladder ran. Taking the rows from it alone hid a phase from every column.
    #[test]
    fn a_phase_the_baseline_cell_missed_still_gets_a_row() {
        let baseline = vec![("parse", 1.0), ("type_check", 2.0)];
        let threaded = vec![("parse", 0.5), ("mlir_passes", 3.0), ("type_check", 1.0)];
        let rows = scaling_rows(&[&baseline, &threaded]);
        assert!(
            rows.contains(&"mlir_passes"),
            "a phase only the threaded cell recorded lost its row: {rows:?}"
        );
    }

    /// Collecting across cells appends, which is not the order the pipeline runs them in. A table
    /// whose rows are out of order reads as a different compile than the one that was measured.
    #[test]
    fn the_rows_stay_in_the_order_the_pipeline_runs_them() {
        let baseline = vec![("type_check", 2.0)];
        let threaded = vec![("parse", 0.5), ("teardown", 1.0), ("macro_expand", 0.2)];
        let rows = scaling_rows(&[&baseline, &threaded]);
        let positions: Vec<usize> = rows.iter().map(|n| phase_order(n)).collect();
        assert!(
            positions.windows(2).all(|w| w[0] <= w[1]),
            "rows are not in pipeline order: {rows:?}"
        );
    }
}
