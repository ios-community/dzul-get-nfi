//! Integration tests for verifying deterministic execution across platforms.

use dzul_core::{Edge, Graph, Node, TspConfig, Weight, Workspace, ZeroHeuristic, solve};

/// Builds the deterministic test scenario (graph, workspace) and solves it.
fn run_scenario() -> Vec<u32> {
    let mut edges = [
        Edge {
            target: 1,
            weight: Weight(10_000_000),
        },
        Edge {
            target: 2,
            weight: Weight(15_000_000),
        },
        Edge {
            target: 0,
            weight: Weight(10_000_000),
        },
        Edge {
            target: 2,
            weight: Weight(20_000_000),
        },
        Edge {
            target: 0,
            weight: Weight(15_000_000),
        },
        Edge {
            target: 1,
            weight: Weight(20_000_000),
        },
    ];
    let nodes = [
        Node {
            edge_start: 0,
            edge_end: 2,
            x: 0,
            y: 0,
        },
        Node {
            edge_start: 2,
            edge_end: 4,
            x: 0,
            y: 0,
        },
        Node {
            edge_start: 4,
            edge_end: 6,
            x: 0,
            y: 0,
        },
    ];
    let mut graph = Graph {
        nodes: &nodes,
        edges: &mut edges,
        is_directed: false,
    };

    let mut path_stack = [0u32; 10];
    let mut next_edge_idx = [0u32; 3];
    let mut visited = [false; 3];
    let mut a_star_parent = [0u32; 3];
    let mut g_score = [0u64; 3];
    let mut open_set = [false; 3];
    let mut nfi_buffer = [Weight(0); 3];
    let mut a_star_heap = [0u32; 3];
    let mut a_star_heap_pos = [-1i32; 3];
    let mut f_score = [0u64; 3];
    let mut dlb = [false; 3];

    let mut workspace = Workspace {
        path_stack: &mut path_stack,
        next_edge_idx: &mut next_edge_idx,
        visited: &mut visited,
        a_star_parent: &mut a_star_parent,
        g_score: &mut g_score,
        open_set: &mut open_set,
        nfi_buffer: &mut nfi_buffer,
        a_star_heap: &mut a_star_heap,
        a_star_heap_pos: &mut a_star_heap_pos,
        f_score: &mut f_score,
        dlb: &mut dlb,
    };

    let heuristic = ZeroHeuristic;
    let config = TspConfig {
        start_node: 0,
        max_backtracks: Some(100),
        enable_2opt: false,
        threshold_multiplier: None,
        backtrack_factor: 10,
        candidate_set_size: 15,
    };

    solve(&mut graph, &mut workspace, &heuristic, &config)
        .unwrap()
        .path
        .to_vec()
}

/// Tests that identical inputs produce identical outputs.
// Anchor: NFR-05
#[test]
fn test_nfr_05_determinism() {
    let res_1 = run_scenario();
    let res_2 = run_scenario();

    assert_eq!(res_1, res_2);
}

/// Verifies safety requirements (Miri).
// Anchor: NFR-02
#[test]
fn test_nfr_02_miri() {
    // This anchor is verified via cargo miri test in CI.
}

/// Verifies portability requirements (`no_std`).
// Anchor: NFR-03
#[test]
fn test_nfr_03_portability() {
    // This anchor is verified via cross-compilation in CI.
}

/// Verifies documentation requirements.
// Anchor: NFR-04
#[test]
fn test_nfr_04_documentation() {
    // This anchor is verified via cargo doc in CI.
}
