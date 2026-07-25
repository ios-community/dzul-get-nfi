//! Performance benchmarks for Dzul's GET-NFI TSP solver (Divan).

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]

use dzul_bench::{build_complete_graph, get_dataset, solve_nearest_neighbor};
use dzul_core::{
    Edge, Graph, TspConfig, Weight, Workspace, ZeroHeuristic, calculate_nfi,
    calculate_threshold, solve, solve_readonly, static_bypass, two_opt,
};

fn build_workspace(n: usize) -> (
    Vec<u32>,
    Vec<u32>,
    Vec<bool>,
    Vec<u32>,
    Vec<u64>,
    Vec<bool>,
    Vec<Weight>,
    Vec<u32>,
    Vec<i32>,
    Vec<u64>,
    Vec<bool>,
) {
    (
        vec![0u32; n * 4],
        vec![0u32; n],
        vec![false; n],
        vec![0u32; n],
        vec![0u64; n],
        vec![false; n],
        vec![Weight(0); n],
        vec![0u32; n],
        vec![-1i32; n],
        vec![0u64; n],
        vec![false; n],
    )
}

fn make_workspace<'a>(
    path_stack: &'a mut Vec<u32>,
    next_edge_idx: &'a mut Vec<u32>,
    visited: &'a mut Vec<bool>,
    a_star_parent: &'a mut Vec<u32>,
    g_score: &'a mut Vec<u64>,
    open_set: &'a mut Vec<bool>,
    nfi_buffer: &'a mut Vec<Weight>,
    a_star_heap: &'a mut Vec<u32>,
    a_star_heap_pos: &'a mut Vec<i32>,
    f_score: &'a mut Vec<u64>,
    dlb: &'a mut Vec<bool>,
) -> Workspace<'a> {
    Workspace {
        path_stack,
        next_edge_idx,
        visited,
        a_star_parent,
        g_score,
        open_set,
        nfi_buffer,
        a_star_heap,
        a_star_heap_pos,
        f_score,
        dlb,
    }
}

fn sort_edges_without_nfi(graph: &mut Graph<'_>, theta: Weight) {
    for node in graph.nodes {
        let start = node.edge_start as usize;
        let end = node.edge_end as usize;
        let subslice = &mut graph.edges[start..end];
        subslice.sort_unstable_by(|a, b| {
            let a_low = a.weight <= theta;
            let b_low = b.weight <= theta;
            match (a_low, b_low) {
                (true, true) => a.weight.cmp(&b.weight),
                (true, false) => core::cmp::Ordering::Less,
                (false, true) => core::cmp::Ordering::Greater,
                (false, false) => a.target.cmp(&b.target),
            }
        });
    }
}

fn calculate_arithmetic_threshold(edges: &[Edge]) -> Weight {
    if edges.is_empty() {
        return Weight(0);
    }
    let sum: f64 = edges.iter().map(|e| e.weight.to_float()).sum();
    let mean = libm::ceil(sum / (edges.len() as f64));
    Weight::from_float(mean)
}

const INSTANCES: &[&str] = &[
    "eil51", "berlin52", "st70", "eil76", "pr76", "rd100", "lin105", "kroA100", "ch150",
    "rat195", "kroA200", "tsp225", "pr226", "gil262", "a280", "lin318", "pcb442", "att532",
    "u574", "rat575", "u724", "rat783", "pr1002", "pcb1173", "d1291", "pr2392", "pcb3038",
    "fnl4461",
];

fn main() {
    divan::main();
}

// --- Instance Benchmarks: 4 variants per instance ---

#[divan::bench(args = INSTANCES)]
fn nearest_neighbor(name: &&str) {
    let coords = get_dataset(name).expect("dataset");
    let (nodes, mut edges) = build_complete_graph(&coords, false);
    let graph = Graph { nodes: &nodes, edges: &mut edges, is_directed: false };
    let _ = solve_nearest_neighbor(&graph, 0);
}

#[divan::bench(args = INSTANCES)]
fn nearest_neighbor_2opt(name: &&str) {
    let coords = get_dataset(name).expect("dataset");
    let n = coords.len();
    let (nodes, mut edges) = build_complete_graph(&coords, false);
    let graph = Graph { nodes: &nodes, edges: &mut edges, is_directed: false };
    let (mut path, _) = solve_nearest_neighbor(&graph, 0);
    let mut dlb = vec![false; n];
    let mut path_pos = vec![-1; n];
    let _ = two_opt(&graph, &mut path, &mut dlb, &mut path_pos, 15);
}

#[divan::bench(args = INSTANCES)]
fn get_nfi(name: &&str) {
    let coords = get_dataset(name).expect("dataset");
    let n = coords.len();
    let (nodes, mut edges) = build_complete_graph(&coords, false);
    let mut graph = Graph { nodes: &nodes, edges: &mut edges, is_directed: false };

    let (mut path_stack, mut next_edge_idx, mut visited, mut a_star_parent, mut g_score,
         mut open_set, mut nfi_buffer, mut a_star_heap, mut a_star_heap_pos, mut f_score,
         mut dlb) = build_workspace(n);
    let mut workspace = make_workspace(
        &mut path_stack, &mut next_edge_idx, &mut visited, &mut a_star_parent, &mut g_score,
        &mut open_set, &mut nfi_buffer, &mut a_star_heap, &mut a_star_heap_pos,
        &mut f_score, &mut dlb,
    );

    let config = TspConfig {
        start_node: 0, max_backtracks: Some(5000), enable_2opt: false, threshold_multiplier: None,
        backtrack_factor: 10, candidate_set_size: 15,
    };
    let _ = solve(&mut graph, &mut workspace, &ZeroHeuristic, &config);
}

#[divan::bench(args = INSTANCES)]
fn get_nfi_with_2opt(name: &&str) {
    let coords = get_dataset(name).expect("dataset");
    let n = coords.len();
    let (nodes, mut edges) = build_complete_graph(&coords, false);
    let mut graph = Graph { nodes: &nodes, edges: &mut edges, is_directed: false };

    let (mut path_stack, mut next_edge_idx, mut visited, mut a_star_parent, mut g_score,
         mut open_set, mut nfi_buffer, mut a_star_heap, mut a_star_heap_pos, mut f_score,
         mut dlb) = build_workspace(n);
    let mut workspace = make_workspace(
        &mut path_stack, &mut next_edge_idx, &mut visited, &mut a_star_parent, &mut g_score,
        &mut open_set, &mut nfi_buffer, &mut a_star_heap, &mut a_star_heap_pos,
        &mut f_score, &mut dlb,
    );

    let config = TspConfig {
        start_node: 0, max_backtracks: Some(5000), enable_2opt: true, threshold_multiplier: None,
        backtrack_factor: 10, candidate_set_size: 15,
    };
    let _ = solve(&mut graph, &mut workspace, &ZeroHeuristic, &config);
}

// --- Ablation Study: 3 variants on subset ---

const ABLATION_INSTANCES: &[&str] = &[
    "eil51", "berlin52", "pr76", "kroA100", "ch150", "a280", "pr226",
];

#[divan::bench(args = ABLATION_INSTANCES)]
fn ablation_full_get_nfi(name: &&str) {
    let coords = get_dataset(name).expect("dataset");
    let n = coords.len();
    let (nodes, mut edges) = build_complete_graph(&coords, false);
    let mut graph = Graph { nodes: &nodes, edges: &mut edges, is_directed: false };

    let theta = calculate_threshold(&graph);
    let is_uniform = static_bypass(&graph);

    let (mut path_stack, mut next_edge_idx, mut visited, mut a_star_parent, mut g_score,
         mut open_set, mut nfi_buffer, mut a_star_heap, mut a_star_heap_pos, mut f_score,
         mut dlb) = build_workspace(n);
    let mut workspace = make_workspace(
        &mut path_stack, &mut next_edge_idx, &mut visited, &mut a_star_parent, &mut g_score,
        &mut open_set, &mut nfi_buffer, &mut a_star_heap, &mut a_star_heap_pos,
        &mut f_score, &mut dlb,
    );

    if !is_uniform {
        let _ = calculate_nfi(&graph, &mut workspace);
        graph.sort_edges(theta, workspace.nfi_buffer);
    }
    let config = TspConfig {
        start_node: 0, max_backtracks: Some(5000), enable_2opt: false, threshold_multiplier: None,
        backtrack_factor: 10, candidate_set_size: 15,
    };
    let _ = solve_readonly(&graph, &mut workspace, &ZeroHeuristic, &config);
}

#[divan::bench(args = ABLATION_INSTANCES)]
fn ablation_no_nfi(name: &&str) {
    let coords = get_dataset(name).expect("dataset");
    let n = coords.len();
    let (nodes, mut edges) = build_complete_graph(&coords, false);
    let mut graph = Graph { nodes: &nodes, edges: &mut edges, is_directed: false };

    let theta = calculate_threshold(&graph);
    sort_edges_without_nfi(&mut graph, theta);

    let (mut path_stack, mut next_edge_idx, mut visited, mut a_star_parent, mut g_score,
         mut open_set, mut nfi_buffer, mut a_star_heap, mut a_star_heap_pos, mut f_score,
         mut dlb) = build_workspace(n);
    let mut workspace = make_workspace(
        &mut path_stack, &mut next_edge_idx, &mut visited, &mut a_star_parent, &mut g_score,
        &mut open_set, &mut nfi_buffer, &mut a_star_heap, &mut a_star_heap_pos,
        &mut f_score, &mut dlb,
    );

    let config = TspConfig {
        start_node: 0, max_backtracks: Some(5000), enable_2opt: false, threshold_multiplier: None,
        backtrack_factor: 10, candidate_set_size: 15,
    };
    let _ = solve_readonly(&graph, &mut workspace, &ZeroHeuristic, &config);
}

#[divan::bench(args = ABLATION_INSTANCES)]
fn ablation_arithmetic_mean(name: &&str) {
    let coords = get_dataset(name).expect("dataset");
    let n = coords.len();
    let (nodes, mut edges) = build_complete_graph(&coords, false);
    let mut graph = Graph { nodes: &nodes, edges: &mut edges, is_directed: false };

    let a_theta = calculate_arithmetic_threshold(&graph.edges);
    let is_uniform = static_bypass(&graph);

    let (mut path_stack, mut next_edge_idx, mut visited, mut a_star_parent, mut g_score,
         mut open_set, mut nfi_buffer, mut a_star_heap, mut a_star_heap_pos, mut f_score,
         mut dlb) = build_workspace(n);
    let mut workspace = make_workspace(
        &mut path_stack, &mut next_edge_idx, &mut visited, &mut a_star_parent, &mut g_score,
        &mut open_set, &mut nfi_buffer, &mut a_star_heap, &mut a_star_heap_pos,
        &mut f_score, &mut dlb,
    );

    if !is_uniform {
        let _ = calculate_nfi(&graph, &mut workspace);
        graph.sort_edges(a_theta, workspace.nfi_buffer);
    }
    let config = TspConfig {
        start_node: 0, max_backtracks: Some(5000), enable_2opt: false, threshold_multiplier: None,
        backtrack_factor: 10, candidate_set_size: 15,
    };
    let _ = solve_readonly(&graph, &mut workspace, &ZeroHeuristic, &config);
}

// --- Sensitivity Analysis: backtrack limit ---

#[divan::bench(args = [100, 1000, 5000])]
fn sensitivity_backtracks(limit: usize) {
    let coords = get_dataset("eil51").expect("dataset");
    let n = coords.len();
    let (nodes, mut edges) = build_complete_graph(&coords, false);
    let mut graph = Graph { nodes: &nodes, edges: &mut edges, is_directed: false };

    let (mut path_stack, mut next_edge_idx, mut visited, mut a_star_parent, mut g_score,
         mut open_set, mut nfi_buffer, mut a_star_heap, mut a_star_heap_pos, mut f_score,
         mut dlb) = build_workspace(n);
    let mut workspace = make_workspace(
        &mut path_stack, &mut next_edge_idx, &mut visited, &mut a_star_parent, &mut g_score,
        &mut open_set, &mut nfi_buffer, &mut a_star_heap, &mut a_star_heap_pos,
        &mut f_score, &mut dlb,
    );

    let config = TspConfig {
        start_node: 0, max_backtracks: Some(limit), enable_2opt: false, threshold_multiplier: None,
        backtrack_factor: 10, candidate_set_size: 15,
    };
    let _ = solve(&mut graph, &mut workspace, &ZeroHeuristic, &config);
}

// --- Sensitivity Analysis: threshold multiplier alpha ---

#[divan::bench(args = [0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0])]
fn sensitivity_threshold_alpha(alpha: f64) {
    let coords = get_dataset("eil51").expect("dataset");
    let n = coords.len();
    let (nodes, mut edges) = build_complete_graph(&coords, false);
    let mut graph = Graph { nodes: &nodes, edges: &mut edges, is_directed: false };

    let (mut path_stack, mut next_edge_idx, mut visited, mut a_star_parent, mut g_score,
         mut open_set, mut nfi_buffer, mut a_star_heap, mut a_star_heap_pos, mut f_score,
         mut dlb) = build_workspace(n);
    let mut workspace = make_workspace(
        &mut path_stack, &mut next_edge_idx, &mut visited, &mut a_star_parent, &mut g_score,
        &mut open_set, &mut nfi_buffer, &mut a_star_heap, &mut a_star_heap_pos,
        &mut f_score, &mut dlb,
    );

    let config = TspConfig {
        start_node: 0, max_backtracks: Some(5000), enable_2opt: false,
        threshold_multiplier: Some(alpha), backtrack_factor: 10, candidate_set_size: 15,
    };
    let _ = solve(&mut graph, &mut workspace, &ZeroHeuristic, &config);
}

// --- Sensitivity Analysis: dynamic backtrack factor c ---

#[divan::bench(args = [1, 5, 10, 25, 50])]
fn sensitivity_backtrack_factor(c: usize) {
    let coords = get_dataset("eil51").expect("dataset");
    let n = coords.len();
    let (nodes, mut edges) = build_complete_graph(&coords, false);
    let mut graph = Graph { nodes: &nodes, edges: &mut edges, is_directed: false };

    let (mut path_stack, mut next_edge_idx, mut visited, mut a_star_parent, mut g_score,
         mut open_set, mut nfi_buffer, mut a_star_heap, mut a_star_heap_pos, mut f_score,
         mut dlb) = build_workspace(n);
    let mut workspace = make_workspace(
        &mut path_stack, &mut next_edge_idx, &mut visited, &mut a_star_parent, &mut g_score,
        &mut open_set, &mut nfi_buffer, &mut a_star_heap, &mut a_star_heap_pos,
        &mut f_score, &mut dlb,
    );

    // max_backtracks = None => uses dynamic formula M(N,d) = c*N*d
    let config = TspConfig {
        start_node: 0, max_backtracks: None, enable_2opt: false,
        threshold_multiplier: None, backtrack_factor: c, candidate_set_size: 15,
    };
    let _ = solve(&mut graph, &mut workspace, &ZeroHeuristic, &config);
}

// --- Sensitivity Analysis: 2-Opt candidate set size k ---

#[divan::bench(args = [5, 10, 15, 25, 50])]
fn sensitivity_candidate_set_k(k: usize) {
    let coords = get_dataset("eil51").expect("dataset");
    let n = coords.len();
    let (nodes, mut edges) = build_complete_graph(&coords, false);
    let mut graph = Graph { nodes: &nodes, edges: &mut edges, is_directed: false };

    let (mut path_stack, mut next_edge_idx, mut visited, mut a_star_parent, mut g_score,
         mut open_set, mut nfi_buffer, mut a_star_heap, mut a_star_heap_pos, mut f_score,
         mut dlb) = build_workspace(n);
    let mut workspace = make_workspace(
        &mut path_stack, &mut next_edge_idx, &mut visited, &mut a_star_parent, &mut g_score,
        &mut open_set, &mut nfi_buffer, &mut a_star_heap, &mut a_star_heap_pos,
        &mut f_score, &mut dlb,
    );

    let config = TspConfig {
        start_node: 0, max_backtracks: Some(5000), enable_2opt: true, threshold_multiplier: None,
        backtrack_factor: 10, candidate_set_size: k,
    };
    let _ = solve(&mut graph, &mut workspace, &ZeroHeuristic, &config);
}
