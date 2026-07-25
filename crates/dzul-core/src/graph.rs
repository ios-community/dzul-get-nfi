//! Compressed Sparse Row (CSR) graph representation.

use crate::{TspError, Weight};

/// Represents an edge in the CSR graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Edge {
    /// The target node index.
    pub target: u32,
    /// The weight of the edge.
    pub weight: Weight,
}

/// Represents a node in the CSR graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Node {
    /// The starting index of outgoing edges in the flat edge slice.
    pub edge_start: u32,
    /// The ending index of outgoing edges in the flat edge slice.
    pub edge_end: u32,
    /// The X coordinate of the node (for Euclidean heuristic).
    pub x: i32,
    /// The Y coordinate of the node (for Euclidean heuristic).
    pub y: i32,
}

/// Represents a Compressed Sparse Row (CSR) graph.
#[derive(Debug)]
pub struct Graph<'a> {
    /// The slice of nodes.
    pub nodes: &'a [Node],
    /// The mutable slice of edges (CSR-ordered).
    pub edges: &'a mut [Edge],
    /// Indicates whether the graph is directed.
    pub is_directed: bool,
}

impl Graph<'_> {
    /// Validates the CSR graph invariants.
    ///
    /// This method checks that the graph structure is consistent and adheres to the
    /// Compressed Sparse Row (CSR) format invariants. Specifically, it verifies that:
    /// - `edge_start <= edge_end` for every node.
    /// - `edge_end` does not exceed the total number of edges.
    /// - The end of one node's edge range matches the start of the next node's edge range.
    /// - All edge targets are valid node indices.
    ///
    /// # Examples
    ///
    /// ```
    /// # use dzul_core::{Graph, Node, Edge, Weight};
    /// let mut edges = [
    ///     Edge { target: 1, weight: Weight(10) },
    ///     Edge { target: 0, weight: Weight(10) },
    /// ];
    /// let nodes = [
    ///     Node { edge_start: 0, edge_end: 1, x: 0, y: 0 },
    ///     Node { edge_start: 1, edge_end: 2, x: 0, y: 0 },
    /// ];
    /// let graph = Graph { nodes: &nodes, edges: &mut edges, is_directed: false };
    /// assert!(graph.validate().is_ok());
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`TspError::InvalidGraph`] if any of the CSR invariants are violated.
    pub fn validate(&self) -> Result<(), TspError> {
        let node_count = self.nodes.len();
        for (i, node) in self.nodes.iter().enumerate() {
            if node.edge_start > node.edge_end {
                return Err(TspError::InvalidGraph);
            }
            if node.edge_end as usize > self.edges.len() {
                return Err(TspError::InvalidGraph);
            }
            if i < node_count - 1 && node.edge_end != self.nodes[i + 1].edge_start {
                return Err(TspError::InvalidGraph);
            }
            for edge in &self.edges[node.edge_start as usize..node.edge_end as usize] {
                if edge.target as usize >= node_count {
                    return Err(TspError::InvalidGraph);
                }
            }
        }
        Ok(())
    }

    /// Sorts the outgoing edges of each node according to the GET-NFI rules.
    ///
    /// This method partitions and sorts the outgoing edges of each node, operating
    /// in-place on the graph's edge slice. Low-Cost edges ($w \le \theta$) are
    /// placed first, sorted ascending by weight. High-Cost edges ($w > \theta$)
    /// are placed second, sorted ascending by target NFI.
    ///
    /// Because the sorted order depends only on $\theta$ and NFI (identical for
    /// every start node), this is typically called once during pre-processing
    /// rather than per search. The search phase itself only reads the edges.
    ///
    /// # Examples
    ///
    /// ```
    /// # use dzul_core::{Graph, Node, Edge, Weight};
    /// let mut edges = [
    ///     Edge { target: 1, weight: Weight(20) },
    ///     Edge { target: 0, weight: Weight(10) },
    /// ];
    /// let nodes = [
    ///     Node { edge_start: 0, edge_end: 2, x: 0, y: 0 },
    /// ];
    /// let mut graph = Graph { nodes: &nodes, edges: &mut edges, is_directed: false };
    /// let nfi = [Weight(100), Weight(200)];
    /// graph.sort_edges(Weight(15), &nfi);
    /// assert_eq!(graph.edges[0].weight, Weight(10));
    /// ```
    pub fn sort_edges(&mut self, theta: Weight, nfi: &[Weight]) {
        for node in self.nodes {
            let start = node.edge_start as usize;
            let end = node.edge_end as usize;
            let subslice = &mut self.edges[start..end];
            subslice.sort_unstable_by(|a, b| {
                let a_low = a.weight <= theta;
                let b_low = b.weight <= theta;
                match (a_low, b_low) {
                    (true, true) => a.weight.cmp(&b.weight),
                    (true, false) => core::cmp::Ordering::Less,
                    (false, true) => core::cmp::Ordering::Greater,
                    (false, false) => {
                        let nfi_a = nfi[a.target as usize];
                        let nfi_b = nfi[b.target as usize];
                        nfi_a.cmp(&nfi_b)
                    }
                }
            });
        }
    }

    /// Checks if the graph is complete.
    ///
    /// A graph is complete if every pair of distinct nodes is connected by a unique edge.
    /// For a directed graph, this means there are $n(n-1)$ edges. For an undirected graph,
    /// there are $n(n-1)/2$ edges.
    ///
    /// # Examples
    ///
    /// ```
    /// # use dzul_core::{Graph, Node, Edge, Weight};
    /// let mut edges = [
    ///     Edge { target: 1, weight: Weight(10) },
    ///     Edge { target: 0, weight: Weight(10) },
    /// ];
    /// let nodes = [
    ///     Node { edge_start: 0, edge_end: 1, x: 0, y: 0 },
    ///     Node { edge_start: 1, edge_end: 2, x: 0, y: 0 },
    /// ];
    /// let graph = Graph { nodes: &nodes, edges: &mut edges, is_directed: false };
    /// assert!(graph.is_complete());
    /// ```
    #[must_use]
    pub fn is_complete(&self) -> bool {
        let n = self.nodes.len();
        if n <= 1 {
            return true;
        }
        let expected_edges = if self.is_directed {
            n * (n - 1)
        } else {
            n * (n - 1) / 2
        };
        self.edges.len() >= expected_edges
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    /// Tests the CSR graph layout validation invariants.
    // Anchor: FR-02
    #[test]
    fn test_fr_02_csr_layout() {
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
        let graph = Graph {
            nodes: &nodes,
            edges: &mut edges,
            is_directed: false,
        };
        assert!(graph.validate().is_ok());

        let nodes_invalid_start = [Node {
            edge_start: 2,
            edge_end: 1,
            x: 0,
            y: 0,
        }];
        let graph_invalid_start = Graph {
            nodes: &nodes_invalid_start,
            edges: &mut edges,
            is_directed: false,
        };
        assert_eq!(graph_invalid_start.validate(), Err(TspError::InvalidGraph));

        let nodes_invalid_end = [Node {
            edge_start: 0,
            edge_end: 3,
            x: 0,
            y: 0,
        }];
        let graph_invalid_end = Graph {
            nodes: &nodes_invalid_end,
            edges: &mut edges,
            is_directed: false,
        };
        assert_eq!(graph_invalid_end.validate(), Err(TspError::InvalidGraph));

        let nodes_non_contiguous = [
            Node {
                edge_start: 0,
                edge_end: 1,
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
        let graph_non_contiguous = Graph {
            nodes: &nodes_non_contiguous,
            edges: &mut edges,
            is_directed: false,
        };
        assert_eq!(graph_non_contiguous.validate(), Err(TspError::InvalidGraph));

        let mut edges_invalid_target = [Edge {
            target: 2,
            weight: Weight(10),
        }];
        let nodes_valid = [
            Node {
                edge_start: 0,
                edge_end: 1,
                x: 0,
                y: 0,
            },
            Node {
                edge_start: 1,
                edge_end: 1,
                x: 0,
                y: 0,
            },
        ];
        let graph_invalid_target = Graph {
            nodes: &nodes_valid,
            edges: &mut edges_invalid_target,
            is_directed: false,
        };
        assert_eq!(graph_invalid_target.validate(), Err(TspError::InvalidGraph));
    }

    /// Tests the edge sorting invariant.
    // Anchor: FR-06
    #[test]
    fn test_fr_06_sorting() {
        let mut edges = [
            Edge {
                target: 1,
                weight: Weight(20),
            },
            Edge {
                target: 2,
                weight: Weight(5),
            },
            Edge {
                target: 3,
                weight: Weight(25),
            },
            Edge {
                target: 0,
                weight: Weight(10),
            },
        ];
        let nodes = [Node {
            edge_start: 0,
            edge_end: 4,
            x: 0,
            y: 0,
        }];
        let nfi = [Weight(100), Weight(400), Weight(300), Weight(200)];
        let mut graph = Graph {
            nodes: &nodes,
            edges: &mut edges,
            is_directed: false,
        };

        graph.sort_edges(Weight(15), &nfi);

        assert_eq!(edges[0].target, 2);
        assert_eq!(edges[1].target, 0);
        assert_eq!(edges[2].target, 3);
        assert_eq!(edges[3].target, 1);
    }

    /// Tests the completeness check of the graph.
    #[test]
    fn test_is_complete() {
        let mut edges_dir = [
            Edge {
                target: 1,
                weight: Weight(1),
            },
            Edge {
                target: 0,
                weight: Weight(1),
            },
        ];
        let nodes_dir = [
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
        let graph_dir = Graph {
            nodes: &nodes_dir,
            edges: &mut edges_dir,
            is_directed: true,
        };
        assert!(graph_dir.is_complete());

        let mut edges_undir = [Edge {
            target: 1,
            weight: Weight(1),
        }];
        let nodes_undir = [
            Node {
                edge_start: 0,
                edge_end: 1,
                x: 0,
                y: 0,
            },
            Node {
                edge_start: 1,
                edge_end: 1,
                x: 0,
                y: 0,
            },
        ];
        let graph_undir = Graph {
            nodes: &nodes_undir,
            edges: &mut edges_undir,
            is_directed: false,
        };
        assert!(graph_undir.is_complete());

        let mut edges_single = [];
        let nodes_single = [Node {
            edge_start: 0,
            edge_end: 0,
            x: 0,
            y: 0,
        }];
        let graph_single = Graph {
            nodes: &nodes_single,
            edges: &mut edges_single,
            is_directed: false,
        };
        assert!(graph_single.is_complete());
    }
}
