//! Statistical analysis for GET-NFI TSP solver — fair apples-to-apples comparisons.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::int_plus_one,
    clippy::identity_op,
    clippy::cast_lossless,
    clippy::needless_borrow,
    clippy::doc_markdown,
    clippy::map_unwrap_or,
    clippy::uninlined_format_args,
    clippy::too_many_lines,
    clippy::similar_names,
    clippy::must_use_candidate
)]
//!
//! # Fair Comparison Structure
//!
//! ## Group 1: Pure Constructive Heuristics (NO 2-Opt)
//! NN vs Farthest Insertion vs Clarke-Wright Savings vs GET-NFI
//!
//! ## Group 2: Constructive + Local Search (WITH 2-Opt)
//! Random+2-Opt vs NN+2-Opt vs FI+2-Opt vs CW+2-Opt vs GET-NFI+2-Opt
//!
//! Reference optimal tour lengths from TSPLIB: <https://www.optsicom.es/tsplib/>

use dzul_bench::{
    DistanceMode, build_complete_graph_with_mode as build_graph, get_dataset,
    solve_clarke_wright_savings, solve_farthest_insertion, solve_nearest_neighbor,
    solve_random_tour,
};
use dzul_core::{
    Edge, Graph, Node, TspConfig, Weight, Workspace, ZeroHeuristic, calculate_nfi,
    calculate_path_cost, calculate_threshold, solve_readonly, static_bypass, two_opt,
};
use std::time::Instant;

/// Known optimal tour lengths for TSPLIB instances (from Concorde/optimal solutions).
fn optimal_cost(name: &str) -> Option<u64> {
    match name {
        "eil51" => Some(426),
        "berlin52" => Some(7_542),
        "st70" => Some(675),
        "eil76" => Some(538),
        "pr76" => Some(108_159),
        "rd100" => Some(7_910),
        "lin105" => Some(14_379),
        "kroA100" => Some(21_282),
        "ch150" => Some(6_528),
        "rat195" => Some(2_323),
        "kroA200" => Some(29_368),
        "tsp225" => Some(3_916),
        "pr226" => Some(80_369),
        "gil262" => Some(2_378),
        "a280" => Some(2_579),
        "lin318" => Some(42_029),
        "pcb442" => Some(50_778),
        "att532" => Some(86_729),
        "u574" => Some(36_905),
        "rat575" => Some(6_773),
        "u724" => Some(41_910),
        "rat783" => Some(8_806),
        "pr1002" => Some(259_045),
        "pcb1173" => Some(56_892),
        "d1291" => Some(50_801),
        "pr2392" => Some(378_032),
        "pcb3038" => Some(137_694),
        "fnl4461" => Some(182_566),
        _ => None,
    }
}

const ALL_INSTANCES: &[&str] = &[
    "eil51", "berlin52", "st70", "eil76", "pr76", "rd100", "lin105", "kroA100", "ch150", "rat195",
    "kroA200", "tsp225", "pr226", "gil262", "a280", "lin318", "pcb442", "att532", "u574", "rat575",
    "u724", "rat783", "pr1002", "pcb1173", "d1291", "pr2392", "pcb3038", "fnl4461",
];

/// Solve GET-NFI with configurable 2-Opt.
fn solve_get_nfi(
    nodes: &[Node],
    edges: &mut [Edge],
    n: usize,
    enable_2opt: bool,
) -> Option<(Vec<u32>, Weight)> {
    let mut graph = Graph {
        nodes,
        edges,
        is_directed: false,
    };
    let is_uniform = static_bypass(&graph);
    let theta = if is_uniform {
        Weight(0)
    } else {
        calculate_threshold(&graph)
    };

    let mut nfi = vec![Weight(0); n];
    let mut g_score = vec![0u64; n];
    let mut open_set = vec![false; n];
    let mut a_star_parent = vec![0u32; n];
    let mut a_star_heap = vec![0u32; n];
    let mut a_star_heap_pos = vec![-1i32; n];
    let mut f_score = vec![0u64; n];
    let mut dlb = vec![false; n];
    let mut next_edge_idx = vec![0u32; n];
    let mut path_stack = vec![0u32; n * 4];
    let mut visited = vec![false; n];

    let mut workspace = Workspace {
        path_stack: &mut path_stack,
        next_edge_idx: &mut next_edge_idx,
        visited: &mut visited,
        a_star_parent: &mut a_star_parent,
        g_score: &mut g_score,
        open_set: &mut open_set,
        nfi_buffer: &mut nfi,
        a_star_heap: &mut a_star_heap,
        a_star_heap_pos: &mut a_star_heap_pos,
        f_score: &mut f_score,
        dlb: &mut dlb,
    };

    if !is_uniform {
        calculate_nfi(&graph, &mut workspace).ok()?;
        graph.sort_edges(theta, workspace.nfi_buffer);
    }

    let config = TspConfig {
        start_node: 0,
        max_backtracks: Some(10_000),
        enable_2opt,
        threshold_multiplier: None,
        backtrack_factor: 10,
        candidate_set_size: 15,
    };
    let result = solve_readonly(&graph, &mut workspace, &ZeroHeuristic, &config).ok()?;
    let cost = calculate_path_cost(&graph, result.path).ok()?;
    Some((result.path.to_vec(), cost))
}

/// Apply 2-Opt to a tour path.
fn apply_2opt(nodes: &[Node], edges: &mut [Edge], n: usize, path: &mut [u32]) -> Weight {
    let graph = Graph {
        nodes,
        edges,
        is_directed: false,
    };
    let mut dlb = vec![false; n];
    let mut path_pos = vec![-1i32; n];
    two_opt(&graph, path, &mut dlb, &mut path_pos, 15).unwrap_or(Weight(0))
}

/// Run NN baseline (returns path, cost).
fn run_nn(nodes: &[Node], edges: &mut [Edge]) -> (Vec<u32>, Weight) {
    let graph = Graph {
        nodes,
        edges,
        is_directed: false,
    };
    solve_nearest_neighbor(&graph, 0)
}

/// Run Farthest Insertion baseline.
fn run_fi(nodes: &[Node], edges: &mut [Edge]) -> (Vec<u32>, Weight) {
    let graph = Graph {
        nodes,
        edges,
        is_directed: false,
    };
    solve_farthest_insertion(&graph, 0)
}

/// Run Clarke-Wright Savings baseline.
fn run_cw(nodes: &[Node], edges: &mut [Edge]) -> (Vec<u32>, Weight) {
    let graph = Graph {
        nodes,
        edges,
        is_directed: false,
    };
    solve_clarke_wright_savings(&graph, 0)
}

/// Run Random Tour baseline.
fn run_random(nodes: &[Node], edges: &mut [Edge], seed: u64) -> (Vec<u32>, Weight) {
    let graph = Graph {
        nodes,
        edges,
        is_directed: false,
    };
    solve_random_tour(&graph, 0, seed)
}

/// Wilcoxon signed-rank test for two paired samples.
fn wilcoxon_signed_rank(a: &[f64], b: &[f64]) -> (f64, f64) {
    assert_eq!(a.len(), b.len());
    let diffs: Vec<f64> = a.iter().zip(b.iter()).map(|(x, y)| x - y).collect();
    let non_zero: Vec<(usize, f64)> = diffs
        .iter()
        .enumerate()
        .filter(|(_, d)| d.abs() > 1e-10)
        .map(|(i, &d)| (i, d))
        .collect();
    let n = non_zero.len();
    if n == 0 {
        return (0.0_f64, 1.0_f64);
    }
    let abs_diffs: Vec<(usize, f64)> = non_zero.iter().map(|&(i, d)| (i, d.abs())).collect();

    let mut sorted = abs_diffs.clone();
    sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && (sorted[j + 1].1 - sorted[j].1).abs() < 1e-12 {
            j += 1;
        }
        let avg_rank = (i + j) as f64 / 2.0 + 1.0;
        for k in i..=j {
            ranks[k] = avg_rank;
        }
        i = j + 1;
    }
    let mut rank_map = std::collections::HashMap::new();
    for (k, &(idx, _)) in sorted.iter().enumerate() {
        rank_map.insert(idx, ranks[k]);
    }

    let mut w_plus = 0.0;
    for (k, &(_, d)) in non_zero.iter().enumerate() {
        let rank = *rank_map.get(&non_zero[k].0).unwrap();
        if d > 0.0 {
            w_plus += rank;
        }
    }
    let w_sum: f64 = (1..=n as u64).sum::<u64>() as f64;
    let w = w_plus.min(w_sum - w_plus);

    let mu = w_sum / 2.0;
    let sigma = libm::sqrt(w_sum * (w_sum + 1.0) / 12.0);
    let z = if sigma > 0.0 { (w - mu) / sigma } else { 0.0 };
    let p_value = 2.0 * (1.0 - normal_cdf(z.abs()));
    (w, p_value)
}

/// Standard normal CDF approximation (Abramowitz and Stegun).
fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / libm::sqrt(2.0)))
}

/// Error function approximation.
fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let y = 1.0
        - (((((1.061_405_429 * t - 0.303_190_491) * t + 0.365_776_303) * t - 0.127_419_418) * t)
            + 0.011_831_088_2)
            * t
            * libm::exp(-x * x);
    sign * y
}

fn stddev(data: &[f64], mean: f64) -> f64 {
    let n = data.len();
    let var = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    libm::sqrt(var)
}

/// Convert Weight to TSPLIB integer.
fn to_int(w: Weight) -> u64 {
    (w.0 + 500_000) / 1_000_000
}

fn gap_pct(cost_int: u64, opt: u64) -> f64 {
    if opt > 0 {
        ((cost_int as f64 - opt as f64) / opt as f64) * 100.0
    } else {
        0.0
    }
}

/// Group 1: Pure Constructive Heuristics (NO 2-Opt).
/// NN vs Farthest Insertion vs Clarke-Wright Savings vs GET-NFI.
#[test]
fn test_group1_pure_constructive() {
    let mut nn_gaps = Vec::new();
    let mut fi_gaps = Vec::new();
    let mut cw_gaps = Vec::new();
    let mut nfi_gaps = Vec::new();
    let mut nfi_better_than_nn = 0usize;

    println!("\n=== GROUP 1: Pure Constructive Heuristics (NO 2-Opt) ===");
    println!(
        "{:<14}{:>15}{:>15}{:>15}{:>15}{:>15}{:>10}{:>10}{:>10}{:>12}",
        "Instance", "Opt", "NN", "FI", "CW", "GET-NFI", "NN%", "FI%", "CW%", "NFI(ms)"
    );
    println!("{}", "-".repeat(131));

    for name in ALL_INSTANCES {
        let Some(coords) = get_dataset(name) else {
            continue;
        };
        let n = coords.len();
        let (nodes, mut edges) = build_graph(&coords, false, DistanceMode::Euc2d);
        let opt = optimal_cost(name).unwrap_or(0);
        if opt == 0 {
            continue;
        }

        let (_nn_path, nn_cost) = run_nn(&nodes, &mut edges);
        let (_fi_path, fi_cost) = run_fi(&nodes, &mut edges);
        let (_cw_path, cw_cost) = run_cw(&nodes, &mut edges);
        let nfi_start = Instant::now();
        let nfi_result = solve_get_nfi(&nodes, &mut edges, n, false);
        let nfi_elapsed_ms = nfi_start.elapsed().as_secs_f64() * 1000.0;

        let nfi_cost = nfi_result.as_ref().map(|c| c.1).unwrap_or(Weight(0));
        let _nfi_path: Vec<u32> = nfi_result.as_ref().map(|c| c.0.clone()).unwrap_or_default();

        let nn_int = to_int(nn_cost);
        let fi_int = to_int(fi_cost);
        let cw_int = to_int(cw_cost);
        let nfi_int = to_int(nfi_cost);

        let nn_gap = gap_pct(nn_int, opt);
        let fi_gap = gap_pct(fi_int, opt);
        let cw_gap = gap_pct(cw_int, opt);
        let nfi_gap = gap_pct(nfi_int, opt);

        println!(
            "{:<14}{:>15}{:>15}{:>15}{:>15}{:>15}{:>9.2}%{:>9.2}%{:>9.2}%{:>12.2}",
            name, opt, nn_int, fi_int, cw_int, nfi_int, nn_gap, fi_gap, cw_gap, nfi_elapsed_ms
        );

        nn_gaps.push(nn_gap);
        fi_gaps.push(fi_gap);
        cw_gaps.push(cw_gap);
        nfi_gaps.push(nfi_gap);
        if nfi_int <= nn_int {
            nfi_better_than_nn += 1;
        }
    }

    let count = nfi_gaps.len();
    assert!(count >= 10, "Need at least 10 instances for Wilcoxon test");

    let mean_nfi = nfi_gaps.iter().sum::<f64>() / count as f64;
    let mean_nn = nn_gaps.iter().sum::<f64>() / count as f64;
    let mean_fi = fi_gaps.iter().sum::<f64>() / count as f64;
    let mean_cw = cw_gaps.iter().sum::<f64>() / count as f64;

    println!("\n{} Summary {}", "=".repeat(20), "=".repeat(20));
    println!("Instances: {count}");
    println!(
        "GET-NFI gap: mean={mean_nfi:.2}%  std={:.2}%",
        stddev(&nfi_gaps, mean_nfi)
    );
    println!(
        "NN gap:      mean={mean_nn:.2}%  std={:.2}%",
        stddev(&nn_gaps, mean_nn)
    );
    println!(
        "FI gap:      mean={mean_fi:.2}%  std={:.2}%",
        stddev(&fi_gaps, mean_fi)
    );
    println!(
        "CW gap:      mean={mean_cw:.2}%  std={:.2}%",
        stddev(&cw_gaps, mean_cw)
    );

    let (w, p) = wilcoxon_signed_rank(&nfi_gaps, &nn_gaps);
    println!("\nWilcoxon (GET-NFI vs NN): W={w:.1}, p={p:.4}");
    let (w, p) = wilcoxon_signed_rank(&nfi_gaps, &fi_gaps);
    println!("Wilcoxon (GET-NFI vs FI): W={w:.1}, p={p:.4}");
    let (w, p) = wilcoxon_signed_rank(&nfi_gaps, &cw_gaps);
    println!("Wilcoxon (GET-NFI vs CW): W={w:.1}, p={p:.4}");

    println!("\nGET-NFI <= NN on {nfi_better_than_nn}/{count} instances");
    assert!(
        nfi_better_than_nn >= count / 3,
        "GET-NFI should be <= NN on at least 1/3 of instances"
    );
}

/// Group 2: Constructive + Local Search (WITH 2-Opt).
/// Random+2-Opt vs NN+2-Opt vs FI+2-Opt vs CW+2-Opt vs GET-NFI+2-Opt.
#[test]
fn test_group2_with_2opt() {
    let mut random_gaps = Vec::new();
    let mut nn_gaps = Vec::new();
    let mut fi_gaps = Vec::new();
    let mut cw_gaps = Vec::new();
    let mut nfi_gaps = Vec::new();
    let mut nfi_better_than_random = 0usize;

    println!("\n=== GROUP 2: Constructive + 2-Opt ===");
    println!(
        "{:<14}{:>15}{:>15}{:>15}{:>15}{:>15}{:>15}{:>10}{:>10}{:>10}{:>12}",
        "Instance",
        "Opt",
        "R+2O",
        "NN+2O",
        "FI+2O",
        "CW+2O",
        "NFI+2O",
        "R%",
        "NN%",
        "NFI%",
        "NFI(ms)"
    );
    println!("{}", "-".repeat(146));

    for name in ALL_INSTANCES {
        let Some(coords) = get_dataset(name) else {
            continue;
        };
        let n = coords.len();
        let (nodes, mut edges) = build_graph(&coords, false, DistanceMode::Euc2d);
        let opt = optimal_cost(name).unwrap_or(0);
        if opt == 0 {
            continue;
        }

        // Random + 2-Opt
        let (mut random_path, _) = run_random(&nodes, &mut edges, 42);
        let random_cost = apply_2opt(&nodes, &mut edges, n, &mut random_path);

        // NN + 2-Opt
        let (mut nn_path, _) = run_nn(&nodes, &mut edges);
        let nn_cost = apply_2opt(&nodes, &mut edges, n, &mut nn_path);

        // FI + 2-Opt
        let (mut fi_path, _) = run_fi(&nodes, &mut edges);
        let fi_cost = apply_2opt(&nodes, &mut edges, n, &mut fi_path);

        // CW + 2-Opt
        let (mut cw_path, _) = run_cw(&nodes, &mut edges);
        let cw_cost = apply_2opt(&nodes, &mut edges, n, &mut cw_path);

        // GET-NFI + 2-Opt
        let nfi_start = Instant::now();
        let nfi_result = solve_get_nfi(&nodes, &mut edges, n, true);
        let nfi_elapsed_ms = nfi_start.elapsed().as_secs_f64() * 1000.0;
        let nfi_cost = nfi_result.as_ref().map(|c| c.1).unwrap_or(Weight(0));

        let r_int = to_int(random_cost);
        let nn_int = to_int(nn_cost);
        let fi_int = to_int(fi_cost);
        let cw_int = to_int(cw_cost);
        let nfi_int = to_int(nfi_cost);

        let r_gap = gap_pct(r_int, opt);
        let nn_gap = gap_pct(nn_int, opt);
        let fi_gap = gap_pct(fi_int, opt);
        let cw_gap = gap_pct(cw_int, opt);
        let nfi_gap = gap_pct(nfi_int, opt);

        println!(
            "{:<14}{:>15}{:>15}{:>15}{:>15}{:>15}{:>15}{:>9.2}%{:>9.2}%{:>9.2}%{:>12}",
            name,
            opt,
            r_int,
            nn_int,
            fi_int,
            cw_int,
            nfi_int,
            r_gap,
            nn_gap,
            nfi_gap,
            nfi_elapsed_ms
        );

        random_gaps.push(r_gap);
        nn_gaps.push(nn_gap);
        fi_gaps.push(fi_gap);
        cw_gaps.push(cw_gap);
        nfi_gaps.push(nfi_gap);
        if nfi_int <= r_int {
            nfi_better_than_random += 1;
        }
    }

    let count = nfi_gaps.len();
    assert!(count >= 10, "Need at least 10 instances for Wilcoxon test");

    let mean_nfi = nfi_gaps.iter().sum::<f64>() / count as f64;
    let mean_random = random_gaps.iter().sum::<f64>() / count as f64;

    println!("\n{} Summary {}", "=".repeat(20), "=".repeat(20));
    println!("Instances: {count}");
    println!(
        "GET-NFI+2Opt gap: mean={mean_nfi:.2}%  std={:.2}%",
        stddev(&nfi_gaps, mean_nfi)
    );
    println!(
        "Random+2Opt gap:  mean={mean_random:.2}%  std={:.2}%",
        stddev(&random_gaps, mean_random)
    );

    let (w, p) = wilcoxon_signed_rank(&nfi_gaps, &random_gaps);
    println!("\nWilcoxon (GET-NFI+2Opt vs Random+2Opt): W={w:.1}, p={p:.4}");
    let (w, p) = wilcoxon_signed_rank(&nfi_gaps, &nn_gaps);
    println!("Wilcoxon (GET-NFI+2Opt vs NN+2Opt): W={w:.1}, p={p:.4}");

    println!("\nGET-NFI+2Opt <= Random+2Opt on {nfi_better_than_random}/{count} instances");
    assert!(
        nfi_better_than_random >= count / 2,
        "GET-NFI+2Opt should be <= Random+2Opt on at least half the instances"
    );
}

/// 2-Opt Ablation: measures initial gap (before 2-Opt) vs final gap (after 2-Opt).
#[test]
fn test_2opt_ablation() {
    let mut initial_gaps = Vec::new();
    let mut final_gaps = Vec::new();
    let mut deltas = Vec::new();

    println!("\n=== 2-Opt Ablation Study ===");
    println!(
        "{:<14}{:>10}{:>12}{:>12}{:>12}",
        "Instance", "Opt", "Initial", "Final", "Delta"
    );
    println!("{}", "-".repeat(60));

    for name in ALL_INSTANCES {
        let Some(coords) = get_dataset(name) else {
            continue;
        };
        let n = coords.len();
        let (nodes, mut edges) = build_graph(&coords, false, DistanceMode::Euc2d);
        let opt = optimal_cost(name).unwrap_or(0);
        if opt == 0 {
            continue;
        }

        // GET-NFI without 2-Opt (initial)
        let initial = solve_get_nfi(&nodes, &mut edges, n, false);
        let initial_cost = initial.as_ref().map(|c| c.1).unwrap_or(Weight(0));
        let initial_int = to_int(initial_cost);
        let initial_gap = gap_pct(initial_int, opt);

        // GET-NFI with 2-Opt (final)
        let final_result = solve_get_nfi(&nodes, &mut edges, n, true);
        let final_cost = final_result.as_ref().map(|c| c.1).unwrap_or(Weight(0));
        let final_int = to_int(final_cost);
        let final_gap = gap_pct(final_int, opt);

        let delta = initial_gap - final_gap;

        println!(
            "{:<14}{:>10}{:>12.2}%{:>12.2}%{:>12.2}%",
            name, opt, initial_gap, final_gap, delta
        );

        initial_gaps.push(initial_gap);
        final_gaps.push(final_gap);
        deltas.push(delta);
    }

    let count = initial_gaps.len();
    assert!(count >= 10, "Need at least 10 instances for ablation");

    let mean_initial = initial_gaps.iter().sum::<f64>() / count as f64;
    let mean_final = final_gaps.iter().sum::<f64>() / count as f64;
    let mean_delta = deltas.iter().sum::<f64>() / count as f64;

    println!("\n{} Summary {}", "=".repeat(20), "=".repeat(20));
    println!("Instances: {count}");
    println!("Initial gap (before 2-Opt): mean={mean_initial:.2}%");
    println!("Final gap (after 2-Opt):    mean={mean_final:.2}%");
    println!("Improvement delta:          mean={mean_delta:.2}%");

    // 2-Opt should improve the tour (delta > 0 means improvement).
    assert!(
        mean_delta > 0.0,
        "2-Opt should improve the initial GET-NFI tour on average"
    );
}
