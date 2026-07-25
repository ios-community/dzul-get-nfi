//! Integration tests for multi-start parallelism.

use dzul_core::{Edge, Graph, Node, TspConfig, Weight, ZeroHeuristic, solve_parallel};

/// Tests multi-start parallelism.
// Anchor: FR-11
#[test]
fn test_fr_11_parallel() {
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

    let graph = Graph {
        nodes: &nodes,
        edges: &mut edges,
        is_directed: false,
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

    let result = solve_parallel(&graph, &heuristic, &config);
    assert!(result.is_ok());
    let res = result.unwrap();
    assert_eq!(res.path.len(), 4);
    assert_eq!(res.total_cost, Weight(45_000_000));
}
