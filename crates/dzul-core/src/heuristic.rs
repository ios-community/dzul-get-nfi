//! A* heuristic interface and implementations.

use crate::graph::Graph;
use crate::weight::Weight;

/// Defines the interface for A* heuristics.
///
/// Implementations of this trait provide an estimate of the cost to travel from
/// one node to another. This estimate is used by the A* algorithm to guide the search.
pub trait Heuristic {
    /// Estimates the cost from node `u` to node `v`.
    ///
    /// This method returns an estimate of the cost to travel from node `u` to node `v`
    /// in the given graph.
    ///
    /// # Examples
    ///
    /// ```
    /// # use dzul_core::{Heuristic, ZeroHeuristic, Graph, Node, Weight};
    /// # let mut edges = [];
    /// # let nodes = [Node { edge_start: 0, edge_end: 0, x: 0, y: 0 }];
    /// # let graph = Graph { nodes: &nodes, edges: &mut edges, is_directed: false };
    /// let heuristic = ZeroHeuristic;
    /// let estimate = heuristic.estimate(0, 0, &graph);
    /// assert_eq!(estimate, Weight(0));
    /// ```
    fn estimate(&self, u: u32, v: u32, graph: &Graph<'_>) -> Weight;
}

/// A heuristic that always returns zero, falling back to Dijkstra's algorithm.
///
/// This heuristic is useful for abstract graphs where coordinates are not available,
/// or when an exact shortest path search is desired without heuristic guidance.
#[derive(Debug, Clone, Copy)]
pub struct ZeroHeuristic;

impl Heuristic for ZeroHeuristic {
    fn estimate(&self, _u: u32, _v: u32, _graph: &Graph<'_>) -> Weight {
        Weight(0)
    }
}

/// A heuristic that computes the scaled Euclidean distance between nodes.
///
/// This heuristic uses the 2D coordinates of the nodes to compute the straight-line
/// distance, which is then scaled to the fixed-point representation.
#[derive(Debug, Clone, Copy)]
pub struct EuclideanHeuristic;

impl Heuristic for EuclideanHeuristic {
    fn estimate(&self, u: u32, v: u32, graph: &Graph<'_>) -> Weight {
        let node_u = &graph.nodes[u as usize];
        let node_v = &graph.nodes[v as usize];
        let dx = f64::from(node_u.x - node_v.x);
        let dy = f64::from(node_u.y - node_v.y);
        let dist = libm::sqrt(dx * dx + dy * dy);
        Weight::from_float(dist)
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use crate::graph::Node;

    /// Tests the heuristic trait and its implementations.
    // Anchor: FR-09
    #[test]
    fn test_fr_09_heuristic() {
        let mut edges = [];
        let nodes = [
            Node {
                edge_start: 0,
                edge_end: 0,
                x: 0,
                y: 0,
            },
            Node {
                edge_start: 0,
                edge_end: 0,
                x: 3,
                y: 4,
            },
        ];
        let graph = Graph {
            nodes: &nodes,
            edges: &mut edges,
            is_directed: false,
        };

        let zero = ZeroHeuristic;
        assert_eq!(zero.estimate(0, 1, &graph), Weight(0));

        let euclidean = EuclideanHeuristic;
        assert_eq!(euclidean.estimate(0, 1, &graph), Weight(5_000_000));
    }
}
