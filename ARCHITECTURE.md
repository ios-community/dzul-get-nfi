# Architecture & Design Specification: Dzul's GET-NFI

This document describes the high-level architecture and design decisions behind Dzul's GET-NFI TSP Solver.

## Module Overview

The project is structured as a decoupled library with a thin CLI binary wrapper. This separation ensures that the core mathematical engine remains independent of I/O, CLI parsing, and benchmarking logic.

```text
+-----------------------------------------------------------------------+
|                              CLI Module                               |
|                           (src/main.rs)                               |
+-----------------------------------+-----------------------------------+
                                    |
                                    v
+-----------------------------------+-----------------------------------+
|                           Dataset Generator                           |
|                           (src/datasets.rs)                           |
+-----------------------------------+-----------------------------------+
                                    |
                                    v
+-----------------------------------+-----------------------------------+
|                           Core ML Engine                              |
|                           (src/solver.rs)                             |
|   +---------------------------------------------------------------+   |
|   |                     GET-NFI Traversal                         |   |
|   |  - Geometric Thresholding                                     |   |
|   |  - Second-Order NFI Preprocessing                             |   |
|   +-------------------------------+-------------------------------+   |
|                                   |                                   |
|                                   v                                   |
|   +---------------------------------------------------------------+   |
|   |                        A* Fallback                            |   |
|   |  - Indexed Binary Heap Priority Queue                         |   |
|   +-------------------------------+-------------------------------+   |
|                                   |                                   |
|                                   v                                   |
|   +---------------------------------------------------------------+   |
|   |                        2-Opt Search                           |   |
|   |  - Candidate-Set-constrained Swaps                            |   |
|   |  - Don't Look Bits (DLB)                                      |   |
|   +---------------------------------------------------------------+   |
+-----------------------------------+-----------------------------------+
                                    |
                                    v
+-----------------------------------+-----------------------------------+
|                          Workspace Module                             |
|                          (src/workspace.rs)                           |
+-----------------------------------------------------------------------+
```

### 1. Core Solver Engine (`src/solver.rs`)
Contains the mathematical representation of the TSP solver.
- **`solve`**: The main entry point for the GET-NFI algorithm. It coordinates thresholding, NFI calculation, edge sorting, backtracking traversal, tour closure, and local search.
- **`two_opt`**: Performs an in-place, zero-allocation 2-Opt local search with Candidate Sets and Don't Look Bits (DLB).
- **`calculate_path_cost`**: Calculates the total cost of a given path.

### 2. Dataset Generator (`src/datasets.rs`)
Provides hardcoded coordinates for standard TSPLIB datasets (`eil51`, `pr76`, `kroA100`, `lin318`, `pcb442`). It includes a deterministic padding mechanism for truncated datasets to keep compilation fast while maintaining full usability.

### 3. Workspace Module (`src/workspace.rs`)
Defines the `Workspace` structure, which holds mutable references to pre-allocated buffers used during execution. This design completely avoids heap allocations during search.

### 4. Heuristic Module (`src/heuristic.rs`)
Defines the `Heuristic` trait and its implementations (`ZeroHeuristic` and `EuclideanHeuristic`) to guide the A* fallback search.

### 5. Weight Module (`src/weight.rs`)
Implements the fixed-point `Weight` representation scaled by $10^6$ to avoid floating-point non-determinism and overhead.

## Key Architectural Decisions

### Memory Management & Allocation
To achieve high performance and meet the strict latency targets, heap allocations are completely avoided inside the search loop.
- The `Workspace` structure is allocated once by the caller and passed as a mutable reference.
- All internal algorithms (including A* fallback and 2-Opt) mutate these buffers in-place.

### Thread Safety & Concurrency
The core solver runs synchronously on a single thread to avoid multi-threading overhead on small graphs. However, the library provides an optional `solve_parallel` function (gated under the `std` feature) that runs the solver from every node in parallel using `rayon`, returning an owned `ParallelTspResult` without any memory leaks.

### Error Handling
All fallible operations return a `Result<T, TspError>`. Performance-critical internal functions use `debug_assert!` or standard assertions to verify array boundaries, avoiding runtime overhead in release builds while maintaining safety.
