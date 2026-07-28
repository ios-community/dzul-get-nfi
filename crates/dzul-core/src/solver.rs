//! Core execution logic for Dzul's GET-NFI TSP solver.
//!
//! # TSP Taxonomy
//!
//! - **Complete Euclidean Graphs (TSPLIB `EUC_2D`):** GET-NFI always yields a
//!   **Strict Hamiltonian Cycle** — every node is visited exactly once and a
//!   direct edge closes the tour (see [`TourType::StrictCycle`]).
//! - **Incomplete/Sparse Graphs:** When no direct closing edge exists, the
//!   A* fallback yields a **Closed Walk** (Graph-TSP / Metric Closure TSP),
//!   where nodes may be revisited along the A* shortest path back to the start
//!   (see [`TourType::ClosedWalk`]).
//!
//! # Paper Title
//!
//! "A Zero-Allocation Dzul's GET-NFI Constructive Heuristic with Candidate-Set
//! 2-Opt for the Travelling Salesperson Problem"

use crate::error::TspError;
use crate::graph::Graph;
use crate::heuristic::Heuristic;
use crate::weight::Weight;
use crate::workspace::Workspace;

/// Defines the configuration for the TSP solver.
#[derive(Debug, Clone, Copy)]
pub struct TspConfig {
    /// The starting node index.
    pub start_node: u32,
    /// The maximum number of backtracks allowed before aborting.
    ///
    /// When `None`, the limit is computed dynamically using
    /// [`calculate_dynamic_backtrack_limit`] with the configured
    /// [`TspConfig::backtrack_factor`].
    pub max_backtracks: Option<usize>,
    /// Enables the in-place 2-Opt local search improvement.
    pub enable_2opt: bool,
    /// An optional multiplier to dynamically scale the geometric threshold.
    pub threshold_multiplier: Option<f64>,
    /// Backtrack factor `c` for the dynamic quadratic limit.
    ///
    /// The dynamic limit is `M(N, d) = c * N * d`, where `N` is the node count
    /// and `d` is the average node degree. For complete graphs, `d = N - 1`,
    /// yielding `M(N) ≈ c * N²`. Default: `10`.
    pub backtrack_factor: usize,
    /// Candidate set size `k` for 2-Opt local search (nearest neighbours).
    ///
    /// Restricts the 2-Opt neighbour search to the top `k` nearest edges per
    /// node. Smaller values are faster but may miss improvements; larger
    /// values explore more at higher cost. Default: `15`.
    pub candidate_set_size: usize,
}

/// Defines the type of tour found.
///
/// # Taxonomy
///
/// On complete Euclidean graphs (e.g. TSPLIB `EUC_2D`), the solver always
/// produces a [`TourType::StrictCycle`] — a Hamiltonian cycle where each node is visited
/// exactly once. On incomplete/sparse graphs, the A* fallback may produce a
/// [`TourType::ClosedWalk`] (Graph-TSP / Metric Closure TSP), where nodes along the A*
/// path back to the start are revisited.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TourType {
    /// A strict Hamiltonian cycle where a direct edge closes the tour.
    StrictCycle,
    /// A closed walk where A* was used to find a path back to the start.
    ClosedWalk,
}

/// Represents the result of a TSP solver execution.
#[derive(Debug)]
pub struct TspResult<'a> {
    /// The path of node indices.
    pub path: &'a [u32],
    /// The total cost of the tour.
    pub total_cost: Weight,
    /// The type of tour found.
    pub tour_type: TourType,
    /// Indicates whether the graph is complete.
    pub is_complete_graph: bool,
}

/// Represents the result of a parallel TSP solver execution.
#[cfg(feature = "parallel")]
#[derive(Debug, Clone)]
pub struct ParallelTspResult {
    /// The path of node indices.
    pub path: Vec<u32>,
    /// The total cost of the tour.
    pub total_cost: Weight,
    /// The type of tour found.
    pub tour_type: TourType,
    /// Indicates whether the graph is complete.
    pub is_complete_graph: bool,
}

#[cfg(feature = "parallel")]
type ParallelResultItem = Result<(Vec<u32>, Weight, TourType), TspError>;

/// Checks if all edge weights in the graph are uniform.
// Anchor: FR-03
#[must_use]
pub fn static_bypass(graph: &Graph<'_>) -> bool {
    if graph.edges.is_empty() {
        return true;
    }

    let first_weight = graph.edges[0].weight;
    for edge in graph.edges.iter() {
        if edge.weight != first_weight {
            return false;
        }
    }
    true
}

/// Calculates the geometric threshold theta.
// Anchor: FR-04
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn calculate_threshold(graph: &Graph<'_>) -> Weight {
    if graph.edges.is_empty() {
        return Weight(0);
    }

    let mut sum_ln = 0.0;
    for edge in graph.edges.iter() {
        let w_f = if edge.weight.0 == 0 {
            1e-6
        } else {
            edge.weight.to_float()
        };
        sum_ln += libm::log(w_f);
    }

    let y_bar = sum_ln / (graph.edges.len() as f64);
    let theta_f = libm::ceil(libm::exp(y_bar));
    Weight::from_float(theta_f)
}

/// Calculates the dynamic quadratic DFS backtrack limit based on graph capacity.
///
/// The formula is `M(N, d) = c * N * d`, where:
/// - `N` = number of nodes
/// - `d` = average node degree (edges per node)
/// - `c` = backtrack factor multiplier
///
/// For complete undirected graphs, `d = N - 1`, yielding `M(N) ≈ c * N²`.
/// This bounds the constructive search depth to guarantee polynomial worst-case
/// execution time O(N³).
///
/// # Examples
///
/// ```
/// # use dzul_core::calculate_dynamic_backtrack_limit;
/// // Complete graph with 100 nodes: c * N * (N-1) = 10 * 100 * 99 = 99000
/// let limit = calculate_dynamic_backtrack_limit(100, 99, 10);
/// assert_eq!(limit, 99_000);
/// ```
#[must_use]
pub fn calculate_dynamic_backtrack_limit(
    node_count: usize,
    avg_degree: usize,
    factor: usize,
) -> usize {
    factor.saturating_mul(node_count).saturating_mul(avg_degree)
}

/// Pre-calculates the Second-Order Node Friendliness Index (NFI) for all nodes.
///
/// # Errors
///
/// Returns [`TspError::ArithmeticOverflow`] if an overflow occurs during NFI calculation.
// Anchor: FR-05
pub fn calculate_nfi(graph: &Graph<'_>, workspace: &mut Workspace<'_>) -> Result<(), TspError> {
    // First pass: Calculate First-Order NFI
    for (i, node) in graph.nodes.iter().enumerate() {
        let mut sum = Weight(0);
        let start = node.edge_start as usize;
        let end = node.edge_end as usize;
        for edge in &graph.edges[start..end] {
            sum = sum
                .checked_add(edge.weight)
                .ok_or(TspError::ArithmeticOverflow)?;
        }
        workspace.nfi_buffer[i] = sum;
    }

    // Second pass: Upgrade to Second-Order NFI (reusing g_score as temporary buffer)
    for i in 0..graph.nodes.len() {
        workspace.g_score[i] = workspace.nfi_buffer[i].0;
    }

    for (i, node) in graph.nodes.iter().enumerate() {
        let mut sum = workspace.g_score[i];
        let start = node.edge_start as usize;
        let end = node.edge_end as usize;
        for edge in &graph.edges[start..end] {
            sum = sum
                .checked_add(workspace.g_score[edge.target as usize])
                .ok_or(TspError::ArithmeticOverflow)?;
        }
        workspace.nfi_buffer[i] = Weight(sum);
    }

    Ok(())
}

/// Solves the TSP on the given graph using Dzul's GET-NFI algorithm.
///
/// This entry point operates on a mutable graph: it performs the one-time
/// GET-NFI pre-processing (threshold calculation, NFI computation, and in-place
/// edge sorting) before running the read-only search. The graph is mutated only
/// to reorder its edges; after sorting it is treated as read-only. Prefer
/// [`solve_readonly`] when you have already pre-sorted the graph or when running
/// [`solve_parallel`], which sorts once and shares a read-only view across threads
/// to avoid duplicating the entire edge list per start node.
///
/// # Errors
///
/// Returns [`TspError::WorkspaceTooSmall`] if the workspace is too small.
/// Returns [`TspError::InvalidGraph`] if the graph is invalid.
/// Returns [`TspError::BacktrackLimitExceeded`] if the backtracking limit is exceeded.
/// Returns [`TspError::NoTourFound`] if no tour is found.
/// Returns [`TspError::ArithmeticOverflow`] if an arithmetic overflow occurs.
pub fn solve<'b, H: Heuristic>(
    graph: &mut Graph<'_>,
    workspace: &'b mut Workspace<'_>,
    heuristic: &H,
    config: &TspConfig,
) -> Result<TspResult<'b>, TspError> {
    debug_assert!(graph.validate().is_ok(), "CSR invariants violated");

    let is_uniform = static_bypass(graph);
    if !is_uniform {
        let mut theta = calculate_threshold(graph);
        if let Some(mult) = config.threshold_multiplier {
            theta = Weight::from_float(theta.to_float() * mult);
        }
        calculate_nfi(graph, workspace)?;
        graph.sort_edges(theta, workspace.nfi_buffer);
    }

    // After pre-processing the graph is only read, so hand a read-only view to
    // the search. This allows the same graph to be shared across threads.
    let readonly: &Graph<'_> = graph;
    solve_readonly(readonly, workspace, heuristic, config)
}

/// Solves the TSP on a pre-processed, read-only graph.
///
/// Unlike [`solve`], this function does **not** mutate the graph. It assumes any
/// required GET-NFI pre-processing (threshold, NFI, edge sorting) has already been
/// applied by the caller. This enables safe sharing of a single graph across
/// multiple threads (see [`solve_parallel`]) without duplicating edge memory.
///
/// # Errors
///
/// Returns [`TspError::WorkspaceTooSmall`] if the workspace is too small.
/// Returns [`TspError::InvalidGraph`] if the graph is invalid.
/// Returns [`TspError::BacktrackLimitExceeded`] if the backtracking limit is exceeded.
/// Returns [`TspError::NoTourFound`] if no tour is found.
/// Returns [`TspError::ArithmeticOverflow`] if an arithmetic overflow occurs.
pub fn solve_readonly<'b, H: Heuristic>(
    graph: &Graph<'_>,
    workspace: &'b mut Workspace<'_>,
    heuristic: &H,
    config: &TspConfig,
) -> Result<TspResult<'b>, TspError> {
    debug_assert!(graph.validate().is_ok(), "CSR invariants violated");
    if !workspace.validate_for_graph(graph.nodes.len()) {
        return Err(TspError::WorkspaceTooSmall);
    }

    let start_node = config.start_node;
    if start_node as usize >= graph.nodes.len() {
        return Err(TspError::InvalidGraph);
    }

    // Compute the backtrack limit: use explicit cap or dynamic quadratic formula.
    let max_backtracks = config.max_backtracks.unwrap_or_else(|| {
        let n = graph.nodes.len();
        let avg_degree = graph.edges.len().checked_div(n).unwrap_or(0);
        calculate_dynamic_backtrack_limit(n, avg_degree, config.backtrack_factor)
    });

    for v in workspace.visited.iter_mut() {
        *v = false;
    }
    for idx in workspace.next_edge_idx.iter_mut() {
        *idx = 0;
    }

    workspace.path_stack[0] = start_node;
    workspace.visited[start_node as usize] = true;
    let mut path_len = 1;
    let mut backtrack_count = 0;

    loop {
        let u = workspace.path_stack[path_len - 1];

        if path_len == graph.nodes.len() {
            match close_tour(graph, workspace, heuristic, config, path_len) {
                Ok((final_len, tour_type)) => {
                    let active_path = &mut workspace.path_stack[0..final_len];
                    let mut total_cost = calculate_path_cost(graph, active_path)?;

                    if config.enable_2opt {
                        total_cost = two_opt(
                            graph,
                            active_path,
                            workspace.dlb,
                            workspace.a_star_heap_pos,
                            config.candidate_set_size,
                        )?;
                    }

                    return Ok(TspResult {
                        path: active_path,
                        total_cost,
                        tour_type,
                        is_complete_graph: graph.is_complete(),
                    });
                }
                Err(TspError::NoTourFound) => {}
                Err(err) => return Err(err),
            }
            if path_len == 1 {
                return Err(TspError::NoTourFound);
            }
            backtrack_count += 1;
            if backtrack_count > max_backtracks {
                return Err(TspError::BacktrackLimitExceeded);
            }
            path_len -= 1;
            workspace.visited[u as usize] = false;
            continue;
        }

        let node_u = &graph.nodes[u as usize];
        let deg = node_u.edge_end - node_u.edge_start;
        let idx = workspace.next_edge_idx[u as usize];

        if idx < deg {
            workspace.next_edge_idx[u as usize] += 1;
            let edge = graph.edges[(node_u.edge_start + idx) as usize];
            let v = edge.target;
            if !workspace.visited[v as usize] {
                workspace.path_stack[path_len] = v;
                workspace.visited[v as usize] = true;
                workspace.next_edge_idx[v as usize] = 0;
                path_len += 1;
            }
        } else {
            // Anchor: FR-08
            if path_len == 1 {
                return Err(TspError::NoTourFound);
            }
            backtrack_count += 1;
            if backtrack_count > max_backtracks {
                return Err(TspError::BacktrackLimitExceeded);
            }
            path_len -= 1;
            workspace.visited[u as usize] = false;
        }
    }
}

// --- Indexed Binary Heap Helpers for A* Fallback ---

fn heap_bubble_up(heap: &mut [u32], pos: &mut [i32], idx: usize, f_scores: &[u64]) {
    let mut i = idx;
    while i > 0 {
        let p = (i - 1) / 2;
        if f_scores[heap[i] as usize] < f_scores[heap[p] as usize] {
            pos.swap(heap[i] as usize, heap[p] as usize);
            heap.swap(i, p);
            i = p;
        } else {
            break;
        }
    }
}

fn heap_bubble_down(heap: &mut [u32], pos: &mut [i32], idx: usize, len: usize, f_scores: &[u64]) {
    let mut i = idx;
    loop {
        let left = 2 * i + 1;
        let right = 2 * i + 2;
        let mut smallest = i;
        if left < len && f_scores[heap[left] as usize] < f_scores[heap[smallest] as usize] {
            smallest = left;
        }
        if right < len && f_scores[heap[right] as usize] < f_scores[heap[smallest] as usize] {
            smallest = right;
        }
        if smallest == i {
            break;
        }
        pos.swap(heap[i] as usize, heap[smallest] as usize);
        heap.swap(i, smallest);
        i = smallest;
    }
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]
fn heap_update(
    heap: &mut [u32],
    pos: &mut [i32],
    len: &mut usize,
    node: u32,
    f_score: u64,
    f_scores: &mut [u64],
) {
    let n = node as usize;
    f_scores[n] = f_score;
    let p = pos[n];
    if p == -1 {
        let idx = *len;
        heap[idx] = node;
        pos[n] = idx as i32;
        *len += 1;
        heap_bubble_up(heap, pos, idx, f_scores);
    } else {
        heap_bubble_up(heap, pos, p as usize, f_scores);
        heap_bubble_down(heap, pos, p as usize, *len, f_scores);
    }
}

fn heap_pop(heap: &mut [u32], pos: &mut [i32], len: &mut usize, f_scores: &[u64]) -> Option<u32> {
    if *len == 0 {
        return None;
    }
    let root = heap[0];
    pos[root as usize] = -1;
    *len -= 1;
    if *len > 0 {
        let last = heap[*len];
        heap[0] = last;
        pos[last as usize] = 0;
        heap_bubble_down(heap, pos, 0, *len, f_scores);
    }
    Some(root)
}

/// Closes the TSP tour by returning to the starting node.
// Anchor: FR-10
fn close_tour<H: Heuristic>(
    graph: &Graph<'_>,
    workspace: &mut Workspace<'_>,
    heuristic: &H,
    config: &TspConfig,
    path_len: usize,
) -> Result<(usize, TourType), TspError> {
    let v_finish = workspace.path_stack[path_len - 1];
    let v_start = config.start_node;

    let node_finish = &graph.nodes[v_finish as usize];
    let mut direct_edge_exists = false;
    for edge in &graph.edges[node_finish.edge_start as usize..node_finish.edge_end as usize] {
        if edge.target == v_start {
            direct_edge_exists = true;
            break;
        }
    }

    if direct_edge_exists {
        if path_len + 1 > workspace.path_stack.len() {
            return Err(TspError::WorkspaceTooSmall);
        }
        workspace.path_stack[path_len] = v_start;
        Ok((path_len + 1, TourType::StrictCycle))
    } else {
        let node_count = graph.nodes.len();
        for i in 0..node_count {
            workspace.g_score[i] = u64::MAX;
            workspace.f_score[i] = u64::MAX;
            workspace.open_set[i] = false;
            workspace.a_star_parent[i] = u32::MAX;
            workspace.a_star_heap_pos[i] = -1;
        }

        workspace.g_score[v_finish as usize] = 0;
        workspace.open_set[v_finish as usize] = true;

        let mut heap_len = 0;
        let h_start = heuristic.estimate(v_finish, v_start, graph).0;
        heap_update(
            workspace.a_star_heap,
            workspace.a_star_heap_pos,
            &mut heap_len,
            v_finish,
            h_start,
            workspace.f_score,
        );

        loop {
            let Some(curr) = heap_pop(
                workspace.a_star_heap,
                workspace.a_star_heap_pos,
                &mut heap_len,
                workspace.f_score,
            ) else {
                return Err(TspError::NoTourFound);
            };

            workspace.open_set[curr as usize] = false;

            if curr == v_start {
                let mut temp = v_start;
                let mut count = 0;
                while temp != v_finish {
                    let p = workspace.a_star_parent[temp as usize];
                    if p == u32::MAX {
                        return Err(TspError::NoTourFound);
                    }
                    temp = p;
                    count += 1;
                }

                if path_len + count > workspace.path_stack.len() {
                    return Err(TspError::WorkspaceTooSmall);
                }

                let mut temp = v_start;
                for i in (0..count).rev() {
                    workspace.path_stack[path_len + i] = temp;
                    temp = workspace.a_star_parent[temp as usize];
                }

                return Ok((path_len + count, TourType::ClosedWalk));
            }

            let node_curr = &graph.nodes[curr as usize];
            for edge in &graph.edges[node_curr.edge_start as usize..node_curr.edge_end as usize] {
                let neighbor = edge.target;
                let weight = edge.weight.0;
                let tentative_g = workspace.g_score[curr as usize].saturating_add(weight);
                if tentative_g < workspace.g_score[neighbor as usize] {
                    workspace.a_star_parent[neighbor as usize] = curr;
                    workspace.g_score[neighbor as usize] = tentative_g;
                    workspace.open_set[neighbor as usize] = true;

                    let h = heuristic.estimate(neighbor, v_start, graph).0;
                    let f = tentative_g.saturating_add(h);
                    heap_update(
                        workspace.a_star_heap,
                        workspace.a_star_heap_pos,
                        &mut heap_len,
                        neighbor,
                        f,
                        workspace.f_score,
                    );
                }
            }
        }
    }
}

fn get_edge_weight(graph: &Graph<'_>, u: u32, v: u32) -> Option<Weight> {
    let node_u = &graph.nodes[u as usize];
    for edge in &graph.edges[node_u.edge_start as usize..node_u.edge_end as usize] {
        if edge.target == v {
            return Some(edge.weight);
        }
    }
    None
}

/// Performs an in-place, zero-allocation 2-Opt local search with Candidate Sets and DLB.
///
/// This method optimizes an existing TSP tour path in-place by iteratively swapping pairs of edges
/// to eliminate crossings. It uses Don't Look Bits (DLB) to prune unproductive searches and restricts
/// swaps to a candidate set of the nearest neighbors for efficiency.
///
/// # Examples
///
/// ```
/// # use dzul_core::{Graph, Node, Edge, Weight, two_opt};
/// let mut edges = [
///     Edge { target: 1, weight: Weight(10_000_000) },
///     Edge { target: 2, weight: Weight(15_000_000) },
///     Edge { target: 0, weight: Weight(10_000_000) },
///     Edge { target: 2, weight: Weight(20_000_000) },
///     Edge { target: 0, weight: Weight(15_000_000) },
///     Edge { target: 1, weight: Weight(20_000_000) },
/// ];
/// let nodes = [
///     Node { edge_start: 0, edge_end: 2, x: 0, y: 0 },
///     Node { edge_start: 2, edge_end: 4, x: 0, y: 0 },
///     Node { edge_start: 4, edge_end: 6, x: 0, y: 0 },
/// ];
/// let graph = Graph { nodes: &nodes, edges: &mut edges, is_directed: false };
///
/// let mut path = [0, 2, 1, 0];
/// let mut dlb = [false; 3];
/// let mut path_pos = [-1; 3];
///
/// let optimized_cost = two_opt(&graph, &mut path, &mut dlb, &mut path_pos, 15).unwrap();
/// assert!(optimized_cost.0 <= 45_000_000);
/// ```
///
/// # Errors
///
/// Returns [`TspError::ArithmeticOverflow`] if an overflow occurs during cost calculation,
/// or [`TspError::InvalidGraph`] if the path contains invalid transitions.
// Anchor: FR-12
#[allow(
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]
pub fn two_opt(
    graph: &Graph<'_>,
    path: &mut [u32],
    dlb: &mut [bool],
    path_pos: &mut [i32],
    candidate_set_size: usize,
) -> Result<Weight, TspError> {
    let n = path.len();
    if n < 4 {
        return calculate_path_cost(graph, path);
    }

    // Initialize Don't Look Bits (DLB) and Path Position Map
    for i in 0..graph.nodes.len() {
        dlb[i] = false;
        path_pos[i] = -1;
    }
    for k in 0..n {
        path_pos[path[k] as usize] = k as i32;
    }

    let mut current_cost = calculate_path_cost(graph, path)?;
    let mut improved = true;
    let mut iterations = 0;

    while improved && iterations < 100 {
        improved = false;
        for i in 1..n - 2 {
            let u = path[i - 1];
            if dlb[u as usize] {
                continue;
            }

            let mut node_improved = false;
            let node_u = &graph.nodes[u as usize];
            let start = node_u.edge_start as usize;
            let end = node_u.edge_end as usize;

            // Candidate Set: Restrict search to top `candidate_set_size` nearest neighbors
            let limit = (end - start).min(candidate_set_size);

            for edge_idx in start..start + limit {
                let v = graph.edges[edge_idx].target;
                let j_pos = path_pos[v as usize];
                if j_pos == -1 {
                    continue;
                }
                let j = j_pos as usize;
                if j <= i || j >= n - 1 {
                    continue;
                }

                if graph.is_directed {
                    // Fallback to full recalculation for Directed Graphs
                    path[i..=j].reverse();
                    let Ok(new_cost) = calculate_path_cost(graph, path) else {
                        path[i..=j].reverse();
                        continue;
                    };
                    if new_cost < current_cost {
                        for k in i..=j {
                            path_pos[path[k] as usize] = k as i32;
                        }
                        current_cost = new_cost;
                        improved = true;
                        node_improved = true;

                        dlb[path[i - 1] as usize] = false;
                        dlb[path[i] as usize] = false;
                        dlb[path[j] as usize] = false;
                        dlb[path[j + 1] as usize] = false;
                        break;
                    }
                    path[i..=j].reverse();
                } else {
                    // O(1) Delta Evaluation for Undirected Graphs
                    if let (
                        Some(w_i_minus_1_i),
                        Some(w_j_j_plus_1),
                        Some(w_i_minus_1_j),
                        Some(w_i_j_plus_1),
                    ) = (
                        get_edge_weight(graph, path[i - 1], path[i]),
                        get_edge_weight(graph, path[j], path[j + 1]),
                        get_edge_weight(graph, path[i - 1], path[j]),
                        get_edge_weight(graph, path[i], path[j + 1]),
                    ) {
                        let gain = w_i_minus_1_j.0.saturating_add(w_i_j_plus_1.0);
                        let loss = w_i_minus_1_i.0.saturating_add(w_j_j_plus_1.0);
                        if gain < loss {
                            let delta = loss - gain;
                            path[i..=j].reverse();

                            // Update positions of reversed elements
                            for k in i..=j {
                                path_pos[path[k] as usize] = k as i32;
                            }

                            current_cost = Weight(
                                current_cost
                                    .0
                                    .checked_sub(delta)
                                    .ok_or(TspError::ArithmeticOverflow)?,
                            );
                            improved = true;
                            node_improved = true;

                            // Reset DLB for modified nodes
                            dlb[path[i - 1] as usize] = false;
                            dlb[path[i] as usize] = false;
                            dlb[path[j] as usize] = false;
                            dlb[path[j + 1] as usize] = false;
                            break;
                        }
                    }
                }
            }

            if !node_improved {
                dlb[u as usize] = true;
            }
        }
        iterations += 1;
    }
    Ok(current_cost)
}

/// Calculates the total cost of the given path.
///
/// This method traverses the path and sums the weights of the edges connecting consecutive nodes.
///
/// # Examples
///
/// ```
/// # use dzul_core::{Graph, Node, Edge, Weight, calculate_path_cost};
/// let mut edges = [
///     Edge { target: 1, weight: Weight(10) },
///     Edge { target: 0, weight: Weight(10) },
/// ];
/// let nodes = [
///     Node { edge_start: 0, edge_end: 1, x: 0, y: 0 },
///     Node { edge_start: 1, edge_end: 2, x: 0, y: 0 },
/// ];
/// let graph = Graph { nodes: &nodes, edges: &mut edges, is_directed: false };
/// let path = [0, 1, 0];
///
/// let cost = calculate_path_cost(&graph, &path).unwrap();
/// assert_eq!(cost, Weight(20));
/// ```
///
/// # Errors
///
/// Returns [`TspError::InvalidGraph`] if any transition in the path does not exist in the graph,
/// or [`TspError::ArithmeticOverflow`] if the sum overflows.
pub fn calculate_path_cost(graph: &Graph<'_>, path: &[u32]) -> Result<Weight, TspError> {
    let mut total_cost = Weight(0);
    for i in 0..path.len() - 1 {
        let u = path[i];
        let v = path[i + 1];
        let node_u = &graph.nodes[u as usize];
        let mut found_weight = None;
        for edge in &graph.edges[node_u.edge_start as usize..node_u.edge_end as usize] {
            if edge.target == v {
                found_weight = Some(edge.weight);
                break;
            }
        }
        let w = found_weight.ok_or(TspError::InvalidGraph)?;
        total_cost = total_cost
            .checked_add(w)
            .ok_or(TspError::ArithmeticOverflow)?;
    }
    Ok(total_cost)
}

/// Runs the solver from every node in parallel using Rayon.
///
/// # Errors
///
/// Returns [`TspError::InvalidGraph`] if the graph is empty or invalid.
/// Returns [`TspError::NoTourFound`] if no tour is found.
/// Returns [`TspError::ArithmeticOverflow`] if an arithmetic overflow occurs.
#[cfg(feature = "parallel")]
pub fn solve_parallel<H: Heuristic + Sync>(
    graph: &Graph<'_>,
    heuristic: &H,
    config: &TspConfig,
) -> Result<ParallelTspResult, TspError> {
    use rayon::prelude::*;

    let node_count = graph.nodes.len();
    if node_count == 0 {
        return Err(TspError::InvalidGraph);
    }

    // Sort the graph ONCE before entering the parallel loop. The edge order
    // depends only on theta and NFI, which are identical for every start node.
    // Cloning the edges once (O(E)) instead of N times (O(N·E)) avoids the memory
    // blow-up on large/complete graphs that previously caused OOM for N ≥ ~800.
    // Sort the graph ONCE before entering the parallel loop. The edge order
    // depends only on theta and NFI, which are identical for every start node.
    // Cloning the edges once (O(E)) instead of N times (O(N·E)) avoids the memory
    // blow-up on large/complete graphs that previously caused OOM for N ≥ ~800.
    let mut sorted_edges = graph.edges.to_vec();
    let mut sorted_graph = Graph {
        nodes: graph.nodes,
        edges: &mut sorted_edges,
        is_directed: graph.is_directed,
    };

    let is_uniform = static_bypass(graph);
    if !is_uniform {
        let mut theta = calculate_threshold(graph);
        if let Some(mult) = config.threshold_multiplier {
            theta = Weight::from_float(theta.to_float() * mult);
        }
        let mut nfi = vec![Weight(0); node_count];
        let mut g_score_temp = vec![0u64; node_count];
        // First pass: first-order NFI
        for (i, node) in sorted_graph.nodes.iter().enumerate() {
            let mut sum = Weight(0);
            for edge in &sorted_graph.edges[node.edge_start as usize..node.edge_end as usize] {
                sum = sum
                    .checked_add(edge.weight)
                    .ok_or(TspError::ArithmeticOverflow)?;
            }
            nfi[i] = sum;
        }
        // Second pass: second-order upgrade
        for i in 0..node_count {
            g_score_temp[i] = nfi[i].0;
        }
        for (i, node) in sorted_graph.nodes.iter().enumerate() {
            let mut acc = g_score_temp[i];
            for edge in &sorted_graph.edges[node.edge_start as usize..node.edge_end as usize] {
                acc = acc
                    .checked_add(g_score_temp[edge.target as usize])
                    .ok_or(TspError::ArithmeticOverflow)?;
            }
            nfi[i] = Weight(acc);
        }
        sorted_graph.sort_edges(theta, &nfi);
    }

    // After sorting, the graph is read-only and can be shared across threads via
    // `&Graph`. Note: `&Graph` with `edges: &mut [Edge]` is `Sync` because
    // `&mut T: Sync` when `T: Sync`; the shared reference prevents exclusive access.

    let results: Vec<ParallelResultItem> = (0..node_count)
        .into_par_iter()
        .map(|start_node| {
            let mut path_stack = vec![0; node_count * crate::workspace::PATH_STACK_MULTIPLIER];
            let mut next_edge_idx = vec![0; node_count];
            let mut visited = vec![false; node_count];
            let mut a_star_parent = vec![0; node_count];
            let mut g_score = vec![0; node_count];
            let mut open_set = vec![false; node_count];
            let mut nfi_buffer = vec![Weight(0); node_count];
            let mut a_star_heap = vec![0; node_count];
            let mut a_star_heap_pos = vec![-1; node_count];
            let mut f_score = vec![0; node_count];
            let mut dlb = vec![false; node_count];

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

            let local_config = TspConfig {
                start_node: u32::try_from(start_node).map_err(|_| TspError::ArithmeticOverflow)?,
                max_backtracks: config.max_backtracks,
                enable_2opt: config.enable_2opt,
                threshold_multiplier: config.threshold_multiplier,
                backtrack_factor: config.backtrack_factor,
                candidate_set_size: config.candidate_set_size,
            };

            let res = solve_readonly(&sorted_graph, &mut workspace, heuristic, &local_config)?;
            Ok((res.path.to_vec(), res.total_cost, res.tour_type))
        })
        .collect();

    let mut best_res: Option<(Vec<u32>, Weight, TourType)> = None;
    for (path, cost, tour_type) in results.into_iter().flatten() {
        if let Some((_, best_cost, _)) = best_res {
            if cost.0 < best_cost.0 {
                best_res = Some((path, cost, tour_type));
            }
        } else {
            best_res = Some((path, cost, tour_type));
        }
    }

    let (best_path, best_cost, best_tour_type) = best_res.ok_or(TspError::NoTourFound)?;

    Ok(ParallelTspResult {
        path: best_path,
        total_cost: best_cost,
        tour_type: best_tour_type,
        is_complete_graph: graph.is_complete(),
    })
}

#[cfg(all(test, feature = "std"))]
#[allow(clippy::too_many_lines)]
mod tests {
    use super::*;
    use crate::graph::{Edge, Node};
    use crate::heuristic::ZeroHeuristic;

    /// Tests the static bypass check.
    // Anchor: FR-03
    #[test]
    fn test_fr_03_bypass() {
        let mut edges_empty = [];
        let nodes_empty = [];
        let graph_empty = Graph {
            nodes: &nodes_empty,
            edges: &mut edges_empty,
            is_directed: false,
        };
        assert!(static_bypass(&graph_empty));

        let mut edges_uniform = [
            Edge {
                target: 1,
                weight: Weight(10),
            },
            Edge {
                target: 0,
                weight: Weight(10),
            },
        ];
        let nodes_uniform = [
            Node {
                edge_start: 0,
                edge_end: 1,
                x: 0,
                y: 0,
            },
            Node {
                edge_start: 1,
                edge_end: 2,
                x: 0,
                y: 0,
            },
        ];
        let graph_uniform = Graph {
            nodes: &nodes_uniform,
            edges: &mut edges_uniform,
            is_directed: false,
        };
        assert!(static_bypass(&graph_uniform));

        let mut edges_non_uniform = [
            Edge {
                target: 1,
                weight: Weight(10),
            },
            Edge {
                target: 0,
                weight: Weight(20),
            },
        ];
        let graph_non_uniform = Graph {
            nodes: &nodes_uniform,
            edges: &mut edges_non_uniform,
            is_directed: false,
        };
        assert!(!static_bypass(&graph_non_uniform));
    }

    /// Tests the geometric threshold calculation.
    // Anchor: FR-04
    #[test]
    fn test_fr_04_threshold() {
        let mut edges_empty = [];
        let nodes_empty = [];
        let graph_empty = Graph {
            nodes: &nodes_empty,
            edges: &mut edges_empty,
            is_directed: false,
        };
        assert_eq!(calculate_threshold(&graph_empty), Weight(0));

        let mut edges_normal = [
            Edge {
                target: 1,
                weight: Weight(1_000_000),
            },
            Edge {
                target: 0,
                weight: Weight(2_000_000),
            },
        ];
        let nodes_normal = [
            Node {
                edge_start: 0,
                edge_end: 1,
                x: 0,
                y: 0,
            },
            Node {
                edge_start: 1,
                edge_end: 2,
                x: 0,
                y: 0,
            },
        ];
        let graph_normal = Graph {
            nodes: &nodes_normal,
            edges: &mut edges_normal,
            is_directed: false,
        };
        let theta = calculate_threshold(&graph_normal);
        assert!(theta.0 > 0);

        let mut edges_zero = [Edge {
            target: 1,
            weight: Weight(0),
        }];
        let nodes_zero = [Node {
            edge_start: 0,
            edge_end: 1,
            x: 0,
            y: 0,
        }];
        let graph_zero = Graph {
            nodes: &nodes_zero,
            edges: &mut edges_zero,
            is_directed: false,
        };
        let theta_zero = calculate_threshold(&graph_zero);
        assert!(theta_zero.0 > 0);
    }

    /// Tests the Node Friendliness Index calculation.
    // Anchor: FR-05
    #[test]
    fn test_fr_05_nfi() {
        let mut edges = [
            Edge {
                target: 1,
                weight: Weight(10),
            },
            Edge {
                target: 0,
                weight: Weight(20),
            },
        ];
        let nodes = [
            Node {
                edge_start: 0,
                edge_end: 1,
                x: 0,
                y: 0,
            },
            Node {
                edge_start: 1,
                edge_end: 2,
                x: 0,
                y: 0,
            },
        ];
        let graph = Graph {
            nodes: &nodes,
            edges: &mut edges,
            is_directed: false,
        };

        let mut path_stack = [0u32; 2];
        let mut next_edge_idx = [0u32; 2];
        let mut visited = [false; 2];
        let mut a_star_parent = [0u32; 2];
        let mut g_score = [0u64; 2];
        let mut open_set = [false; 2];
        let mut nfi_buffer = [Weight(0); 2];
        let mut a_star_heap = [0u32; 2];
        let mut a_star_heap_pos = [-1i32; 2];
        let mut f_score = [0u64; 2];
        let mut dlb = [false; 2];

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

        assert!(calculate_nfi(&graph, &mut workspace).is_ok());
        assert_eq!(workspace.nfi_buffer[0], Weight(30)); // Second-Order NFI: 10 + 20
        assert_eq!(workspace.nfi_buffer[1], Weight(30)); // Second-Order NFI: 20 + 10

        let mut edges_overflow = [
            Edge {
                target: 1,
                weight: Weight(u64::MAX),
            },
            Edge {
                target: 1,
                weight: Weight(1),
            },
        ];
        let nodes_overflow = [Node {
            edge_start: 0,
            edge_end: 2,
            x: 0,
            y: 0,
        }];
        let graph_overflow = Graph {
            nodes: &nodes_overflow,
            edges: &mut edges_overflow,
            is_directed: false,
        };
        let mut nfi_buffer_overflow = [Weight(0); 1];
        let mut workspace_overflow = Workspace {
            path_stack: &mut path_stack,
            next_edge_idx: &mut next_edge_idx,
            visited: &mut visited,
            a_star_parent: &mut a_star_parent,
            g_score: &mut g_score,
            open_set: &mut open_set,
            nfi_buffer: &mut nfi_buffer_overflow,
            a_star_heap: &mut a_star_heap,
            a_star_heap_pos: &mut a_star_heap_pos,
            f_score: &mut f_score,
            dlb: &mut dlb,
        };
        assert_eq!(
            calculate_nfi(&graph_overflow, &mut workspace_overflow),
            Err(TspError::ArithmeticOverflow)
        );
    }

    /// Tests the backtracking behavior of the solver.
    // Anchor: FR-08
    #[test]
    fn test_fr_08_backtrack() {
        let mut edges = [
            Edge {
                target: 1,
                weight: Weight(10),
            },
            Edge {
                target: 2,
                weight: Weight(10),
            },
            Edge {
                target: 0,
                weight: Weight(10),
            },
            Edge {
                target: 3,
                weight: Weight(10),
            },
            Edge {
                target: 1,
                weight: Weight(10),
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
                edge_end: 3,
                x: 0,
                y: 0,
            },
            Node {
                edge_start: 3,
                edge_end: 4,
                x: 0,
                y: 0,
            },
            Node {
                edge_start: 4,
                edge_end: 5,
                x: 0,
                y: 0,
            },
        ];
        let mut graph = Graph {
            nodes: &nodes,
            edges: &mut edges,
            is_directed: true,
        };

        let mut path_stack = [0u32; 10];
        let mut next_edge_idx = [0u32; 4];
        let mut visited = [false; 4];
        let mut a_star_parent = [0u32; 4];
        let mut g_score = [0u64; 4];
        let mut open_set = [false; 4];
        let mut nfi_buffer = [Weight(0); 4];
        let mut a_star_heap = [0u32; 4];
        let mut a_star_heap_pos = [-1i32; 4];
        let mut f_score = [0u64; 4];
        let mut dlb = [false; 4];

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
            max_backtracks: Some(10),
            enable_2opt: false,
            threshold_multiplier: None,
            backtrack_factor: 10,
            candidate_set_size: 15,
        };

        {
            let result = solve(&mut graph, &mut workspace, &heuristic, &config);
            assert!(result.is_ok());
            let res = result.unwrap();
            assert_eq!(res.path, &[0, 2, 3, 1, 0]);
            assert_eq!(res.tour_type, TourType::StrictCycle);
        }

        let config_limit = TspConfig {
            start_node: 0,
            max_backtracks: Some(0),
            enable_2opt: false,
            threshold_multiplier: None,
            backtrack_factor: 10,
            candidate_set_size: 15,
        };
        let result_limit = solve(&mut graph, &mut workspace, &heuristic, &config_limit);
        assert_eq!(result_limit.unwrap_err(), TspError::BacktrackLimitExceeded);
    }

    /// Tests the tour closure logic (`StrictCycle` vs `ClosedWalk`).
    // Anchor: FR-10
    #[test]
    fn test_fr_10_closure() {
        let mut edges_strict = [
            Edge {
                target: 1,
                weight: Weight(10),
            },
            Edge {
                target: 0,
                weight: Weight(10),
            },
        ];
        let nodes_strict = [
            Node {
                edge_start: 0,
                edge_end: 1,
                x: 0,
                y: 0,
            },
            Node {
                edge_start: 1,
                edge_end: 2,
                x: 0,
                y: 0,
            },
        ];
        let mut graph_strict = Graph {
            nodes: &nodes_strict,
            edges: &mut edges_strict,
            is_directed: false,
        };

        let mut path_stack = [0u32; 10];
        let mut next_edge_idx = [0u32; 2];
        let mut visited = [false; 2];
        let mut a_star_parent = [0u32; 2];
        let mut g_score = [0u64; 2];
        let mut open_set = [false; 2];
        let mut nfi_buffer = [Weight(0); 2];
        let mut a_star_heap = [0u32; 2];
        let mut a_star_heap_pos = [-1i32; 2];
        let mut f_score = [0u64; 2];
        let mut dlb = [false; 2];

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
            max_backtracks: None,
            enable_2opt: false,
            threshold_multiplier: None,
            backtrack_factor: 10,
            candidate_set_size: 15,
        };

        let res_strict = solve(&mut graph_strict, &mut workspace, &heuristic, &config).unwrap();
        assert_eq!(res_strict.tour_type, TourType::StrictCycle);
        assert_eq!(res_strict.path, &[0, 1, 0]);

        let mut edges_closed = [
            Edge {
                target: 1,
                weight: Weight(10),
            },
            Edge {
                target: 2,
                weight: Weight(10),
            },
            Edge {
                target: 0,
                weight: Weight(10),
            },
            Edge {
                target: 1,
                weight: Weight(10),
            },
        ];
        let nodes_closed = [
            Node {
                edge_start: 0,
                edge_end: 1,
                x: 0,
                y: 0,
            },
            Node {
                edge_start: 1,
                edge_end: 3,
                x: 0,
                y: 0,
            },
            Node {
                edge_start: 3,
                edge_end: 4,
                x: 0,
                y: 0,
            },
        ];
        let mut graph_closed = Graph {
            nodes: &nodes_closed,
            edges: &mut edges_closed,
            is_directed: true,
        };

        let mut path_stack_closed = [0u32; 10];
        let mut next_edge_idx_closed = [0u32; 3];
        let mut visited_closed = [false; 3];
        let mut a_star_parent_closed = [0u32; 3];
        let mut g_score_closed = [0u64; 3];
        let mut open_set_closed = [false; 3];
        let mut nfi_buffer_closed = [Weight(0); 3];
        let mut a_star_heap_closed = [0u32; 3];
        let mut a_star_heap_pos_closed = [-1i32; 3];
        let mut f_score_closed = [0u64; 3];
        let mut dlb_closed = [false; 3];

        let mut workspace_closed = Workspace {
            path_stack: &mut path_stack_closed,
            next_edge_idx: &mut next_edge_idx_closed,
            visited: &mut visited_closed,
            a_star_parent: &mut a_star_parent_closed,
            g_score: &mut g_score_closed,
            open_set: &mut open_set_closed,
            nfi_buffer: &mut nfi_buffer_closed,
            a_star_heap: &mut a_star_heap_closed,
            a_star_heap_pos: &mut a_star_heap_pos_closed,
            f_score: &mut f_score_closed,
            dlb: &mut dlb_closed,
        };

        let config_closed = TspConfig {
            start_node: 0,
            max_backtracks: Some(usize::MAX),
            enable_2opt: false,
            threshold_multiplier: None,
            backtrack_factor: 10,
            candidate_set_size: 15,
        };

        let res_closed = solve(
            &mut graph_closed,
            &mut workspace_closed,
            &heuristic,
            &config_closed,
        )
        .unwrap();
        assert_eq!(res_closed.tour_type, TourType::ClosedWalk);
        assert_eq!(res_closed.path, &[0, 1, 2, 1, 0]);

        let mut edges_fail = [
            Edge {
                target: 1,
                weight: Weight(10),
            },
            Edge {
                target: 2,
                weight: Weight(10),
            },
        ];
        let nodes_fail = [
            Node {
                edge_start: 0,
                edge_end: 1,
                x: 0,
                y: 0,
            },
            Node {
                edge_start: 1,
                edge_end: 2,
                x: 0,
                y: 0,
            },
            Node {
                edge_start: 2,
                edge_end: 2,
                x: 0,
                y: 0,
            },
        ];
        let mut graph_fail = Graph {
            nodes: &nodes_fail,
            edges: &mut edges_fail,
            is_directed: true,
        };

        let mut path_stack_fail = [0u32; 10];
        let mut next_edge_idx_fail = [0u32; 3];
        let mut visited_fail = [false; 3];
        let mut a_star_parent_fail = [0u32; 3];
        let mut g_score_fail = [0u64; 3];
        let mut open_set_fail = [false; 3];
        let mut nfi_buffer_fail = [Weight(0); 3];
        let mut a_star_heap_fail = [0u32; 3];
        let mut a_star_heap_pos_fail = [-1i32; 3];
        let mut f_score_fail = [0u64; 3];
        let mut dlb_fail = [false; 3];

        let mut workspace_fail = Workspace {
            path_stack: &mut path_stack_fail,
            next_edge_idx: &mut next_edge_idx_fail,
            visited: &mut visited_fail,
            a_star_parent: &mut a_star_parent_fail,
            g_score: &mut g_score_fail,
            open_set: &mut open_set_fail,
            nfi_buffer: &mut nfi_buffer_fail,
            a_star_heap: &mut a_star_heap_fail,
            a_star_heap_pos: &mut a_star_heap_pos_fail,
            f_score: &mut f_score_fail,
            dlb: &mut dlb_fail,
        };

        let res_fail = solve(
            &mut graph_fail,
            &mut workspace_fail,
            &heuristic,
            &config_closed,
        );
        assert_eq!(res_fail.unwrap_err(), TspError::NoTourFound);

        let mut path_stack_small = [0u32; 3];
        let mut workspace_small = Workspace {
            path_stack: &mut path_stack_small,
            next_edge_idx: &mut next_edge_idx_closed,
            visited: &mut visited_closed,
            a_star_parent: &mut a_star_parent_closed,
            g_score: &mut g_score_closed,
            open_set: &mut open_set_closed,
            nfi_buffer: &mut nfi_buffer_closed,
            a_star_heap: &mut a_star_heap_closed,
            a_star_heap_pos: &mut a_star_heap_pos_closed,
            f_score: &mut f_score_closed,
            dlb: &mut dlb_closed,
        };
        let mut graph_closed_2 = Graph {
            nodes: &nodes_closed,
            edges: &mut edges_closed,
            is_directed: true,
        };
        let res_small = solve(
            &mut graph_closed_2,
            &mut workspace_small,
            &heuristic,
            &config_closed,
        );
        assert_eq!(res_small.unwrap_err(), TspError::WorkspaceTooSmall);
    }

    /// Tests various error conditions in the solver.
    #[test]
    fn test_solve_errors() {
        let mut edges = [
            Edge {
                target: 1,
                weight: Weight(10),
            },
            Edge {
                target: 0,
                weight: Weight(10),
            },
        ];
        let nodes = [
            Node {
                edge_start: 0,
                edge_end: 1,
                x: 0,
                y: 0,
            },
            Node {
                edge_start: 1,
                edge_end: 2,
                x: 0,
                y: 0,
            },
        ];

        let mut graph = Graph {
            nodes: &nodes,
            edges: &mut edges,
            is_directed: false,
        };

        let mut next_edge_idx = [0u32; 2];
        let mut visited = [false; 2];
        let mut a_star_parent = [0u32; 2];
        let mut g_score = [0u64; 2];
        let mut open_set = [false; 2];
        let mut nfi_buffer = [Weight(0); 2];
        let mut a_star_heap = [0u32; 2];
        let mut a_star_heap_pos = [-1i32; 2];
        let mut f_score = [0u64; 2];
        let mut dlb = [false; 2];

        let heuristic = ZeroHeuristic;
        let config = TspConfig {
            start_node: 0,
            max_backtracks: None,
            enable_2opt: false,
            threshold_multiplier: None,
            backtrack_factor: 10,
            candidate_set_size: 15,
        };

        {
            let mut path_stack = [0u32; 1];
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

            assert_eq!(
                solve(&mut graph, &mut workspace, &heuristic, &config).unwrap_err(),
                TspError::WorkspaceTooSmall
            );
        }

        let mut path_stack_ok = [0u32; 5];
        let mut workspace_ok = Workspace {
            path_stack: &mut path_stack_ok,
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
        let config_oob = TspConfig {
            start_node: 2,
            max_backtracks: None,
            enable_2opt: false,
            threshold_multiplier: None,
            backtrack_factor: 10,
            candidate_set_size: 15,
        };
        assert_eq!(
            solve(&mut graph, &mut workspace_ok, &heuristic, &config_oob).unwrap_err(),
            TspError::InvalidGraph
        );
    }

    /// Tests the 2-Opt local search improvement.
    // Anchor: FR-12
    #[test]
    fn test_fr_12_2opt() {
        let mut edges = [
            Edge {
                target: 1,
                weight: Weight(10),
            },
            Edge {
                target: 2,
                weight: Weight(15),
            },
            Edge {
                target: 3,
                weight: Weight(20),
            },
            Edge {
                target: 0,
                weight: Weight(10),
            },
            Edge {
                target: 2,
                weight: Weight(35),
            },
            Edge {
                target: 3,
                weight: Weight(25),
            },
            Edge {
                target: 0,
                weight: Weight(15),
            },
            Edge {
                target: 1,
                weight: Weight(35),
            },
            Edge {
                target: 3,
                weight: Weight(30),
            },
            Edge {
                target: 0,
                weight: Weight(20),
            },
            Edge {
                target: 1,
                weight: Weight(25),
            },
            Edge {
                target: 2,
                weight: Weight(30),
            },
        ];
        let nodes = [
            Node {
                edge_start: 0,
                edge_end: 3,
                x: 0,
                y: 0,
            },
            Node {
                edge_start: 3,
                edge_end: 6,
                x: 0,
                y: 0,
            },
            Node {
                edge_start: 6,
                edge_end: 9,
                x: 0,
                y: 0,
            },
            Node {
                edge_start: 9,
                edge_end: 12,
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
        let mut next_edge_idx = [0u32; 4];
        let mut visited = [false; 4];
        let mut a_star_parent = [0u32; 4];
        let mut g_score = [0u64; 4];
        let mut open_set = [false; 4];
        let mut nfi_buffer = [Weight(0); 4];
        let mut a_star_heap = [0u32; 4];
        let mut a_star_heap_pos = [-1i32; 4];
        let mut f_score = [0u64; 4];
        let mut dlb = [false; 4];

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

        let config_no_2opt = TspConfig {
            start_node: 0,
            max_backtracks: None,
            enable_2opt: false,
            threshold_multiplier: None,
            backtrack_factor: 10,
            candidate_set_size: 15,
        };
        let cost_no_2opt = solve(&mut graph, &mut workspace, &ZeroHeuristic, &config_no_2opt)
            .unwrap()
            .total_cost;

        let config_2opt = TspConfig {
            start_node: 0,
            max_backtracks: None,
            enable_2opt: true,
            threshold_multiplier: None,
            backtrack_factor: 10,
            candidate_set_size: 15,
        };
        let res_2opt = solve(&mut graph, &mut workspace, &ZeroHeuristic, &config_2opt).unwrap();

        assert!(res_2opt.total_cost.0 <= cost_no_2opt.0);
    }

    /// Tests the dynamic threshold multiplier.
    // Anchor: FR-13
    #[test]
    fn test_fr_13_threshold_multiplier() {
        let mut edges = [
            Edge {
                target: 1,
                weight: Weight(10),
            },
            Edge {
                target: 0,
                weight: Weight(20),
            },
        ];
        let nodes = [
            Node {
                edge_start: 0,
                edge_end: 1,
                x: 0,
                y: 0,
            },
            Node {
                edge_start: 1,
                edge_end: 2,
                x: 0,
                y: 0,
            },
        ];
        let mut graph = Graph {
            nodes: &nodes,
            edges: &mut edges,
            is_directed: false,
        };

        let mut path_stack = [0u32; 5];
        let mut next_edge_idx = [0u32; 2];
        let mut visited = [false; 2];
        let mut a_star_parent = [0u32; 2];
        let mut g_score = [0u64; 2];
        let mut open_set = [false; 2];
        let mut nfi_buffer = [Weight(0); 2];
        let mut a_star_heap = [0u32; 2];
        let mut a_star_heap_pos = [-1i32; 2];
        let mut f_score = [0u64; 2];
        let mut dlb = [false; 2];

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
            max_backtracks: None,
            enable_2opt: false,
            threshold_multiplier: Some(1.5),
            backtrack_factor: 10,
            candidate_set_size: 15,
        };

        let res = solve(&mut graph, &mut workspace, &ZeroHeuristic, &config);
        assert!(res.is_ok());
    }

    /// Tests the dynamic quadratic backtrack limit formula M(N,d) = c*N*d.
    // Anchor: FR-15
    #[test]
    fn test_fr_15_dynamic_backtrack_limit() {
        // Complete graph: N=100, d=99, c=10 => 99_000
        assert_eq!(calculate_dynamic_backtrack_limit(100, 99, 10), 99_000);
        // Sparse graph: N=50, d=5, c=10 => 2_500
        assert_eq!(calculate_dynamic_backtrack_limit(50, 5, 10), 2_500);
        // Saturating on overflow: huge values clamp to usize::MAX
        assert_eq!(
            calculate_dynamic_backtrack_limit(usize::MAX, usize::MAX, 2),
            usize::MAX
        );
        // Factor scaling: c=1 vs c=10
        assert_eq!(calculate_dynamic_backtrack_limit(10, 9, 1), 90);
        assert_eq!(calculate_dynamic_backtrack_limit(10, 9, 10), 900);
    }
}
