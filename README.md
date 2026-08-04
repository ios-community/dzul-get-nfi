# GET-NFI TSP Solver

> **A Zero-Allocation GET-NFI Constructive Heuristic with Candidate-Set 2-Opt for the Travelling Salesperson Problem**

A zero-allocation, `no_std` compatible Travelling Salesperson Problem (TSP) solver written from scratch in Rust, featuring GET-NFI heuristic algorithm.

[![License](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](#license)
[![Rust](https://img.shields.io/badge/rust-1.96.0%2B-orange.svg)](https://github.com)

## Overview

`dzul-get-nfi` is a zero-unsafe, single-threaded neural-heuristic TSP solver designed for both complete/incomplete and directed/undirected graphs. It achieves deterministic execution with exactly zero heap allocations during search, making it suitable for bare-metal embedded systems and high-performance computing environments.

The core algorithm utilizes a geometric thresholding mechanism to partition edges into low-cost and high-cost sets, combined with a Second-Order Node Friendliness Index (NFI) to prioritize isolated nodes. For incomplete graphs, an $O(E \log V)$ Indexed Binary Heap A* fallback is used to guarantee tour closure.

## Features

- **Second-Order Node Friendliness Index (NFI)** \
Advanced local connectivity metric to prioritize isolated nodes during traversal.
- **Candidate-Set-constrained 2-Opt** \
Local search restricted to the top $k$-nearest neighbors, reducing complexity from $O(N^2)$ to $O(N)$.
- **Don't Look Bits (DLB)** \
Intelligent node-flagging to bypass unproductive local search swaps.
- **Indexed Binary Heap A\* Fallback** \
$O(E \log V)$ priority queue for tour closure on incomplete graphs.
- **Dynamic Quadratic DFS Backtrack Limit** \
Bounds search depth with $M(N, d) = c \cdot N \cdot d$, guaranteeing polynomial worst-case $O(N^3)$ time.
- **TSPLIB EUC_2D Integer Rounding** \
Exact per-edge nearest-integer rounding matches Concorde benchmark optimal values without drift.
- **Zero Heap Allocation** \
Fully compatible with `no_std` and bare-metal environments.
- **Multi-Start Parallelism** \
Optional parallel execution from every node using `rayon` (gated under the `std` feature).

## TSP Taxonomy

- **Complete Euclidean Graphs (TSPLIB EUC_2D):** GET-NFI always yields a
  **Strict Hamiltonian Cycle** — every node is visited exactly once and a
  direct edge closes the tour.
- **Incomplete/Sparse Graphs:** When no direct closing edge exists, the A*
  fallback yields a **Closed Walk** (Graph-TSP / Metric Closure TSP), where
  nodes along the A* shortest path back to the start may be revisited.

## Installation

Ensure you have Rust 1.96.0 or newer installed. Clone the repository and build the project:

```bash
git clone https://github.com/ios-community/dzul-get-nfi.git
cd dzul-get-nfi
cargo build --release
```

## Usage

### Command Line Interface (CLI)

Run the benchmark binary to solve a specific TSPLIB instance. For example, to solve `eil51` with 2-Opt enabled:

```bash
cargo run --release -- \
  --instance eil51 \
  --sparsity 1.0 \
  --2opt \
  --backtracks 5000
```

#### CLI Arguments

- `--instance <NAME>`: TSPLIB instance name (see [Benchmark Suite](#benchmark-suite) for full list) [default: `eil51`].
- `--sparsity <RATIO>`: Edge keep ratio for incomplete graphs (0.0 to 1.0) [default: 1.0].
- `--2opt`: Enable the in-place 2-Opt local search improvement.
- `--backtracks <LIMIT>`: Maximum backtracking limit [default: 5000].
- `--directed`: Simulate an Asymmetric TSP (ATSP) by scaling directed edges.

### Running Benchmarks

```bash
# Run full benchmark suite (28 datasets)
cargo bench --features std

# Run ablation study only
cargo bench --features std --bench solver_benches -- ablation

# Run statistical analysis (optimality gap, Wilcoxon test)
cargo test --features std --test statistics -- --nocapture
```

#### Dataset Caching

Large datasets (n > 500; `pr1002`, `pcb1173`, `d1291`, `pr2392`, `pcb3038`, `fnl4461` etc.) are **not** hardcoded in the source. On first use they are downloaded from the TSPLIB mirror at `http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/` and cached to `datasets/{name}.tsp`. Subsequent runs use the local cache. If network access is unavailable, these large instances will be skipped by the benchmark/tests.

#### Benchmark Suite

The solver is evaluated on 28 TSPLIB instances across three scales:

| Scale | n | Instances |
|---|---|---|
| Small | <100 | `eil51`, `berlin52`, `st70`, `eil76`, `pr76` |
| Medium | 100--500 | `rd100`, `lin105`, `kroA100`, `ch150`, `rat195`, `kroA200`, `tsp225`, `pr226`, `gil262`, `a280`, `lin318`, `pcb442`, `att532` |
| Large | >500 | `u574`, `rat575`, `u724`, `rat783`, `pr1002`, `pcb1173`, `d1291`, `pr2392`, `pcb3038`, `fnl4461` |

### Library Usage

You can also use `dzul-get-nfi` as a library in your own Rust projects. Add it to your `Cargo.toml` dependencies, then use the API:

```rust
use dzul_get_nfi::{Graph, Node, Edge, Weight, Workspace, ZeroHeuristic, TspConfig, solve};

fn main() -> Result<(), dzul_get_nfi::TspError> {
    let mut edges = [
        Edge { target: 1, weight: Weight(10_000_000) },
        Edge { target: 2, weight: Weight(15_000_000) },
        Edge { target: 0, weight: Weight(10_000_000) },
        Edge { target: 2, weight: Weight(20_000_000) },
        Edge { target: 0, weight: Weight(15_000_000) },
        Edge { target: 1, weight: Weight(20_000_000) },
    ];

    let nodes = [
        Node { edge_start: 0, edge_end: 2, x: 0, y: 0 },
        Node { edge_start: 2, edge_end: 4, x: 0, y: 0 },
        Node { edge_start: 4, edge_end: 6, x: 0, y: 0 },
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

    let config = TspConfig {
        start_node: 0,
        max_backtracks: Some(100),
        enable_2opt: true,
        threshold_multiplier: None,
        backtrack_factor: 10,
        candidate_set_size: 15,
    };

    let result = solve(&mut graph, &mut workspace, &ZeroHeuristic, &config)?;
    println!("Tour Cost: {}", result.total_cost.to_float());
    println!("Tour Path: {:?}", result.path);
    Ok(())
}
```

## Feature Gating

The multi-start parallel solver is optional and can be disabled to compile the engine in minimal or bare-metal environments without standard library dependencies.

- **`std` (Enabled by default):** Pulls in the `rayon` dependency to enable parallel multi-start execution.

To compile for bare-metal (`no_std`) environments:

```bash
cargo build --release --no-default-features --target thumbv7m-none-eabi
```

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.



