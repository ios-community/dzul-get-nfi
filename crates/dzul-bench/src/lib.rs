//! CLI runner, dataset fetcher, and benchmarks for dzul-core.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::must_use_candidate,
    clippy::missing_panics_doc,
    clippy::too_many_lines
)]

pub mod dataset;

use dzul_core::{Edge, Node, Weight};

pub use dataset::get_dataset;
pub use dataset::get_synthetic_dataset;

/// Distance computation mode for graph construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceMode {
    /// Fixed-point mode: distances scaled by 10^6 without per-edge rounding.
    FixedPoint,
    /// TSPLIB `EUC_2D` mode: nearest-integer rounding per edge (`nint`).
    ///
    /// Produces exact integer distances matching Concorde TSPLIB benchmark
    /// optimal values without floating-point drift.
    Euc2d,
}

impl DistanceMode {
    /// Computes the edge weight between two coordinates using this mode.
    #[must_use]
    pub fn compute(self, x1: f64, y1: f64, x2: f64, y2: f64) -> Weight {
        match self {
            Self::FixedPoint => {
                let dx = x1 - x2;
                let dy = y1 - y2;
                Weight::from_float(libm::sqrt(dx * dx + dy * dy))
            }
            Self::Euc2d => Weight::euc_2d(x1, y1, x2, y2),
        }
    }
}

/// Builds a complete Euclidean graph from coordinates.
pub fn build_complete_graph(coords: &[(f64, f64)], is_directed: bool) -> (Vec<Node>, Vec<Edge>) {
    build_complete_graph_with_mode(coords, is_directed, DistanceMode::FixedPoint)
}

/// Builds a complete Euclidean graph using the specified distance mode.
pub fn build_complete_graph_with_mode(
    coords: &[(f64, f64)],
    is_directed: bool,
    mode: DistanceMode,
) -> (Vec<Node>, Vec<Edge>) {
    let n = coords.len();
    let mut nodes = Vec::with_capacity(n);
    let mut edges = Vec::with_capacity(n * (n - 1));

    let mut edge_idx = 0;
    for i in 0..n {
        let start = edge_idx;
        for j in 0..n {
            if i != j {
                let mut weight = mode.compute(coords[i].0, coords[i].1, coords[j].0, coords[j].1);

                if is_directed && i < j {
                    weight = Weight((weight.0 as f64 * 1.2) as u64);
                }

                edges.push(Edge {
                    target: j as u32,
                    weight,
                });
                edge_idx += 1;
            }
        }
        nodes.push(Node {
            edge_start: start as u32,
            edge_end: edge_idx as u32,
            x: coords[i].0 as i32,
            y: coords[i].1 as i32,
        });
    }
    (nodes, edges)
}

/// Builds an incomplete (sparse) graph by keeping only the nearest neighbors.
pub fn build_incomplete_graph(
    coords: &[(f64, f64)],
    keep_ratio: f64,
    is_directed: bool,
) -> (Vec<Node>, Vec<Edge>) {
    build_incomplete_graph_with_mode(coords, keep_ratio, is_directed, DistanceMode::FixedPoint)
}

/// Builds an incomplete (sparse) graph using the specified distance mode.
pub fn build_incomplete_graph_with_mode(
    coords: &[(f64, f64)],
    keep_ratio: f64,
    is_directed: bool,
    mode: DistanceMode,
) -> (Vec<Node>, Vec<Edge>) {
    let n = coords.len();
    let mut nodes = Vec::with_capacity(n);
    let mut all_edges = Vec::with_capacity(n);

    for i in 0..n {
        let mut node_edges = Vec::with_capacity(n - 1);
        for j in 0..n {
            if i != j {
                let mut weight = mode.compute(coords[i].0, coords[i].1, coords[j].0, coords[j].1);

                if is_directed && i < j {
                    weight = Weight((weight.0 as f64 * 1.2) as u64);
                }

                node_edges.push(Edge {
                    target: j as u32,
                    weight,
                });
            }
        }
        node_edges.sort_unstable_by_key(|a| a.weight);
        let keep_count = ((node_edges.len() as f64) * keep_ratio).round() as usize;
        let keep_count = keep_count.max(1);
        node_edges.truncate(keep_count);
        all_edges.push(node_edges);
    }

    let mut flat_edges = Vec::new();
    let mut edge_idx = 0;
    for i in 0..n {
        let start = edge_idx;
        for edge in &all_edges[i] {
            flat_edges.push(*edge);
            edge_idx += 1;
        }
        nodes.push(Node {
            edge_start: start as u32,
            edge_end: edge_idx as u32,
            x: coords[i].0 as i32,
            y: coords[i].1 as i32,
        });
    }

    (nodes, flat_edges)
}

/// Verifies a tour contains exactly `N + 1` elements and covers all unique
/// node indices `0..N` (start node appears twice, all others once).
// Anchor: FR-17
fn assert_tour_integrity(tour: &[u32], n: usize) {
    assert_eq!(
        tour.len(),
        n + 1,
        "tour length {} != N+1 ({})",
        tour.len(),
        n + 1
    );
    let mut seen = vec![false; n];
    for &node in &tour[..n] {
        assert!((node as usize) < n, "node index {node} out of range 0..{n}");
        assert!(!seen[node as usize], "duplicate node {node} in tour");
        seen[node as usize] = true;
    }
    assert!(
        seen.iter().all(|&v| v),
        "tour does not cover all nodes 0..{n}"
    );
}

/// Nearest Neighbor baseline heuristic.
pub fn solve_nearest_neighbor(graph: &dzul_core::Graph<'_>, start_node: u32) -> (Vec<u32>, Weight) {
    let n = graph.nodes.len();
    let mut path = Vec::with_capacity(n + 1);
    let mut visited = vec![false; n];

    let mut curr = start_node;
    path.push(curr);
    visited[curr as usize] = true;

    let mut total_cost = Weight(0);

    for _ in 1..n {
        let node = &graph.nodes[curr as usize];
        let mut min_dist = u64::MAX;
        let mut next_node = None;
        let mut next_weight = Weight(0);

        for edge in &graph.edges[node.edge_start as usize..node.edge_end as usize] {
            if !visited[edge.target as usize] && edge.weight.0 < min_dist {
                min_dist = edge.weight.0;
                next_node = Some(edge.target);
                next_weight = edge.weight;
            }
        }

        if let Some(next) = next_node {
            curr = next;
            path.push(curr);
            visited[curr as usize] = true;
            total_cost = total_cost.checked_add(next_weight).unwrap();
        } else {
            break;
        }
    }

    let node = &graph.nodes[curr as usize];
    for edge in &graph.edges[node.edge_start as usize..node.edge_end as usize] {
        if edge.target == start_node {
            path.push(start_node);
            total_cost = total_cost.checked_add(edge.weight).unwrap();
            break;
        }
    }

    (path, total_cost)
}

/// Farthest Insertion heuristic.
///
/// Starting from a trivial tour (a single node), repeatedly inserts the
/// unvisited node farthest from any node already in the tour at the position
/// that minimises the insertion cost. Runs in O(N²) time.
pub fn solve_farthest_insertion(
    graph: &dzul_core::Graph<'_>,
    start_node: u32,
) -> (Vec<u32>, Weight) {
    let n = graph.nodes.len();
    if n == 0 {
        return (Vec::new(), Weight(0));
    }

    // Helper closure for O(1) average edge weight lookup
    let get_w = |u: usize, v: usize| -> u64 {
        let node = &graph.nodes[u];
        for edge in &graph.edges[node.edge_start as usize..node.edge_end as usize] {
            if edge.target == v as u32 {
                return edge.weight.0;
            }
        }
        u64::MAX
    };

    let mut tour: Vec<u32> = vec![start_node, start_node];
    let mut in_tour = vec![false; n];
    in_tour[start_node as usize] = true;

    // Array tracking minimum distance from unvisited nodes to the active tour
    let mut min_dist_to_tour = vec![u64::MAX; n];
    for i in 0..n {
        if i as u32 != start_node {
            min_dist_to_tour[i] = get_w(start_node as usize, i);
        }
    }

    while tour.len() <= n {
        // Find the unvisited node farthest from any node in the current tour in O(N)
        let mut farthest_node = None;
        let mut max_dist = 0u64;

        for i in 0..n {
            if !in_tour[i] && min_dist_to_tour[i] > max_dist {
                max_dist = min_dist_to_tour[i];
                farthest_node = Some(i as u32);
            }
        }

        let Some(insert_node) = farthest_node else {
            break;
        };

        // Find best insertion position minimizing insertion detour cost
        let mut best_pos = 1;
        let mut min_delta = u64::MAX;
        for k in 0..tour.len() - 1 {
            let a = tour[k] as usize;
            let b = tour[k + 1] as usize;
            let ins = insert_node as usize;

            let w_a_n = get_w(a, ins);
            let w_n_b = get_w(ins, b);
            let w_a_b = get_w(a, b);

            let delta = w_a_n.saturating_add(w_n_b).saturating_sub(w_a_b);
            if delta < min_delta {
                min_delta = delta;
                best_pos = k + 1;
            }
        }

        tour.insert(best_pos, insert_node);
        in_tour[insert_node as usize] = true;

        // Update min_dist_to_tour array for remaining unvisited nodes in O(N)
        let ins = insert_node as usize;
        for i in 0..n {
            if !in_tour[i] {
                let d = get_w(i, ins);
                if d < min_dist_to_tour[i] {
                    min_dist_to_tour[i] = d;
                }
            }
        }
    }

    assert_tour_integrity(&tour, n);
    let total_cost = dzul_core::calculate_path_cost(graph, &tour).unwrap_or(Weight(0));
    (tour, total_cost)
}

/// Clarke-Wright Savings algorithm.
///
/// Builds an initial set of routes `0 -> i -> 0` then iteratively merges them
/// by largest savings until a single tour remains. Runs in O(N² log N) time.
pub fn solve_clarke_wright_savings(
    graph: &dzul_core::Graph<'_>,
    start_node: u32,
) -> (Vec<u32>, Weight) {
    let n = graph.nodes.len();
    if n == 0 {
        return (Vec::new(), Weight(0));
    }
    if n == 1 {
        return (vec![start_node, start_node], Weight(0));
    }

    let get_w = |u: usize, v: usize| -> u64 {
        let node = &graph.nodes[u];
        for edge in &graph.edges[node.edge_start as usize..node.edge_end as usize] {
            if edge.target == v as u32 {
                return edge.weight.0;
            }
        }
        u64::MAX
    };

    // Calculate Savings: s(i, j) = d(depot, i) + d(depot, j) - d(i, j)
    let mut savings: Vec<(u64, u32, u32)> = Vec::with_capacity(n * (n - 1) / 2);
    let s_node = start_node as usize;
    for i in 0..n {
        if i == s_node {
            continue;
        }
        let d0i = get_w(s_node, i);
        if d0i == u64::MAX {
            continue;
        }
        for j in (i + 1)..n {
            if j == s_node {
                continue;
            }
            let d0j = get_w(s_node, j);
            let dij = get_w(i, j);
            if d0j == u64::MAX || dij == u64::MAX {
                continue;
            }
            let s = d0i.saturating_add(d0j).saturating_sub(dij);
            savings.push((s, i as u32, j as u32));
        }
    }
    savings.sort_unstable_by(|a, b| b.0.cmp(&a.0));

    // Union-Find (Disjoint-Set) data structure to track route components
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(p: &mut [usize], i: usize) -> usize {
        let mut root = i;
        while root != p[root] {
            root = p[root];
        }
        let mut curr = i;
        while curr != root {
            let nxt = p[curr];
            p[curr] = root;
            curr = nxt;
        }
        root
    }

    let mut next_node: Vec<u32> = vec![u32::MAX; n];
    let mut prev_node: Vec<u32> = vec![u32::MAX; n];
    let mut degree: Vec<usize> = vec![0; n];

    for &(_s, u_u32, v_u32) in &savings {
        let u = u_u32 as usize;
        let v = v_u32 as usize;

        // Nodes in a linear path can have at most degree 2
        if degree[u] >= 2 || degree[v] >= 2 {
            continue;
        }

        // Prevent cycle formation using Union-Find
        if find(&mut parent, u) == find(&mut parent, v) {
            continue;
        }

        // Link u -> v or v -> u
        if next_node[u] == u32::MAX && prev_node[v] == u32::MAX {
            next_node[u] = v_u32;
            prev_node[v] = u_u32;
            degree[u] += 1;
            degree[v] += 1;
            let root_u = find(&mut parent, u);
            let root_v = find(&mut parent, v);
            parent[root_u] = root_v;
        } else if next_node[v] == u32::MAX && prev_node[u] == u32::MAX {
            next_node[v] = u_u32;
            prev_node[u] = v_u32;
            degree[u] += 1;
            degree[v] += 1;
            let root_u = find(&mut parent, u);
            let root_v = find(&mut parent, v);
            parent[root_u] = root_v;
        }
    }

    // Reconstruct continuous tour from depot
    let mut head = u32::MAX;
    for i in 0..n {
        if i as u32 != start_node && prev_node[i] == u32::MAX {
            head = i as u32;
            break;
        }
    }

    let mut tour: Vec<u32> = Vec::with_capacity(n + 1);
    tour.push(start_node);

    if head != u32::MAX {
        let mut curr = head;
        while curr != u32::MAX && tour.len() <= n {
            tour.push(curr);
            curr = next_node[curr as usize];
        }
    }

    // Append remaining unconnected nodes
    if tour.len() < n + 1 {
        let mut visited = vec![false; n];
        for &node in &tour {
            visited[node as usize] = true;
        }
        for i in 0..n {
            if !visited[i] {
                tour.push(i as u32);
            }
        }
    }
    tour.push(start_node);

    assert_tour_integrity(&tour, n);
    let total_cost = dzul_core::calculate_path_cost(graph, &tour).unwrap_or(Weight(0));
    (tour, total_cost)
}

/// Random tour generator.
///
/// Produces a random permutation of nodes starting and ending at `start_node`.
/// Uses a simple Fisher-Yates shuffle.
pub fn solve_random_tour(
    graph: &dzul_core::Graph<'_>,
    start_node: u32,
    seed: u64,
) -> (Vec<u32>, Weight) {
    let n = graph.nodes.len();
    if n == 0 {
        return (Vec::new(), Weight(0));
    }

    // LCG PRNG for reproducibility (no extra deps).
    let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut rng = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        state
    };

    let mut nodes: Vec<u32> = (0..n as u32).filter(|&x| x != start_node).collect();
    for i in (1..nodes.len()).rev() {
        let j = (rng() as usize) % (i + 1);
        nodes.swap(i, j);
    }

    let mut tour = Vec::with_capacity(n + 1);
    tour.push(start_node);
    tour.extend_from_slice(&nodes);
    tour.push(start_node);

    assert_tour_integrity(&tour, n);
    let total_cost = dzul_core::calculate_path_cost(graph, &tour).unwrap_or(Weight(0));
    (tour, total_cost)
}

/// Looks up the weight of the edge `u -> v` in the graph.
fn edge_weight(graph: &dzul_core::Graph<'_>, u: u32, v: u32) -> Option<Weight> {
    let node_u = &graph.nodes[u as usize];
    for edge in &graph.edges[node_u.edge_start as usize..node_u.edge_end as usize] {
        if edge.target == v {
            return Some(edge.weight);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use dzul_core::{Edge, Graph, Node};

    fn build_complete_4_node_graph() -> (Vec<Node>, Vec<Edge>) {
        let coords: &[(f64, f64)] = &[(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)];
        build_complete_graph(coords, false)
    }

    /// Tests the expanded baseline suite produces valid tours (N+1 elements,
    /// all nodes covered 0..N) via tour integrity assertions.
    // Anchor: FR-17
    #[test]
    fn test_fr_17_baseline_suite() {
        let (nodes, mut edges) = build_complete_4_node_graph();
        let graph = Graph {
            nodes: &nodes,
            edges: &mut edges,
            is_directed: false,
        };
        let n = graph.nodes.len();

        // Farthest Insertion: integrity asserted inside.
        let (fi_tour, _) = solve_farthest_insertion(&graph, 0);
        assert_eq!(fi_tour.len(), n + 1);

        // Clarke-Wright Savings: integrity asserted inside.
        let (cw_tour, _) = solve_clarke_wright_savings(&graph, 0);
        assert_eq!(cw_tour.len(), n + 1);

        // Random Tour: integrity asserted inside.
        let (rand_tour, _) = solve_random_tour(&graph, 0, 42);
        assert_eq!(rand_tour.len(), n + 1);
    }
}
