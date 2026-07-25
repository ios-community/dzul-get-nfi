//! Integration tests for zero heap allocation during solver execution.

use dzul_core::{Edge, Graph, Node, TspConfig, Weight, Workspace, ZeroHeuristic, solve};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

struct TrackingAllocator;

static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, Ordering::SeqCst);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static A: TrackingAllocator = TrackingAllocator;

/// Verifies that exactly zero bytes are allocated on the heap during solver execution.
// Anchor: NFR-01
#[test]
fn test_nfr_01_zero_alloc() {
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

    let allocs_before = ALLOC_COUNT.load(Ordering::SeqCst);

    let result = solve(&mut graph, &mut workspace, &heuristic, &config);

    let allocs_after = ALLOC_COUNT.load(Ordering::SeqCst);

    assert!(result.is_ok(), "Solver failed to find a tour");
    assert_eq!(
        allocs_before, allocs_after,
        "Memory allocations detected during solve execution! (Before: {allocs_before}, After: {allocs_after})",
    );
}
