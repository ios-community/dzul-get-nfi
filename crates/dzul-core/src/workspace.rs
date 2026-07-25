//! Pre-allocated execution state workspace.

use crate::weight::Weight;

/// Recommended multiplier for `path_stack` length when solving incomplete graphs.
///
/// On incomplete (sparse) graphs, the tour may close via an A* fallback path that
/// revisits nodes (a *Closed Walk*), making the final path longer than `node_count`.
/// Allocating `path_stack` with length `node_count * PATH_STACK_MULTIPLIER` is
/// sufficient for all cases encountered by the solver.
///
/// For complete graphs, `node_count + 1` is sufficient (a strict Hamiltonian cycle
/// plus the closing return to start). For incomplete graphs, `node_count * 2` is
/// recommended, and `node_count * 4` provides extra headroom.
///
/// This multiplier is already applied internally by [`solve_parallel`](crate::solver::solve_parallel).
/// Users of [`solve`](crate::solver::solve) or [`solve_readonly`](crate::solver::solve_readonly)
/// should use this constant to size their `path_stack` when working with incomplete graphs.
pub const PATH_STACK_MULTIPLIER: usize = 4;

/// Pre-allocated workspace for zero-allocation execution.
///
/// This structure holds mutable references to pre-allocated buffers that are used
/// during the execution of the TSP solver. This design avoids dynamic memory allocation
/// (heap allocation) during the search, making it suitable for `no_std` and bare-metal
/// environments.
#[derive(Debug)]
pub struct Workspace<'a> {
    /// Buffer for the active path stack.
    pub path_stack: &'a mut [u32],
    /// Buffer for tracking the next edge index to explore per node.
    pub next_edge_idx: &'a mut [u32],
    /// Buffer for tracking visited nodes.
    pub visited: &'a mut [bool],
    /// Buffer for A* parent pointers.
    pub a_star_parent: &'a mut [u32],
    /// Buffer for A* g-scores.
    pub g_score: &'a mut [u64],
    /// Buffer for A* open set membership.
    pub open_set: &'a mut [bool],
    /// Buffer for storing calculated Node Friendliness Indices (NFI).
    pub nfi_buffer: &'a mut [Weight],
    /// Buffer for A* indexed binary heap.
    pub a_star_heap: &'a mut [u32],
    /// Buffer for A* indexed binary heap positions.
    pub a_star_heap_pos: &'a mut [i32],
    /// Buffer for A* f-scores.
    pub f_score: &'a mut [u64],
    /// Buffer for Don't Look Bits (DLB) in 2-Opt.
    pub dlb: &'a mut [bool],
}

impl Workspace<'_> {
    /// Validates that the workspace buffers are large enough for the given node count.
    ///
    /// All buffers except `path_stack` need length >= `node_count`. `path_stack` must
    /// hold the full tour, which may exceed `node_count` on incomplete graphs due to
    /// A* fallback closure (see [`PATH_STACK_MULTIPLIER`]).
    ///
    /// # Examples
    ///
    /// ```
    /// # use dzul_core::{Workspace, Weight};
    /// let mut path_stack = [0u32; 5];
    /// let mut next_edge_idx = [0u32; 5];
    /// let mut visited = [false; 5];
    /// let mut a_star_parent = [0u32; 5];
    /// let mut g_score = [0u64; 5];
    /// let mut open_set = [false; 5];
    /// let mut nfi_buffer = [Weight(0); 5];
    /// let mut a_star_heap = [0u32; 5];
    /// let mut a_star_heap_pos = [0i32; 5];
    /// let mut f_score = [0u64; 5];
    /// let mut dlb = [false; 5];
    ///
    /// let workspace = Workspace {
    ///     path_stack: &mut path_stack,
    ///     next_edge_idx: &mut next_edge_idx,
    ///     visited: &mut visited,
    ///     a_star_parent: &mut a_star_parent,
    ///     g_score: &mut g_score,
    ///     open_set: &mut open_set,
    ///     nfi_buffer: &mut nfi_buffer,
    ///     a_star_heap: &mut a_star_heap,
    ///     a_star_heap_pos: &mut a_star_heap_pos,
    ///     f_score: &mut f_score,
    ///     dlb: &mut dlb,
    /// };
    ///
    /// assert!(workspace.validate_for_graph(5));
    /// assert!(!workspace.validate_for_graph(6));
    /// ```
    #[must_use]
    pub fn validate_for_graph(&self, node_count: usize) -> bool {
        self.path_stack.len() >= node_count
            && self.next_edge_idx.len() >= node_count
            && self.visited.len() >= node_count
            && self.a_star_parent.len() >= node_count
            && self.g_score.len() >= node_count
            && self.open_set.len() >= node_count
            && self.nfi_buffer.len() >= node_count
            && self.a_star_heap.len() >= node_count
            && self.a_star_heap_pos.len() >= node_count
            && self.f_score.len() >= node_count
            && self.dlb.len() >= node_count
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    /// Tests the workspace validation and sizing invariants.
    // Anchor: FR-07
    #[test]
    fn test_fr_07_workspace() {
        let mut path_stack = [0u32; 5];
        let mut next_edge_idx = [0u32; 5];
        let mut visited = [false; 5];
        let mut a_star_parent = [0u32; 5];
        let mut g_score = [0u64; 5];
        let mut open_set = [false; 5];
        let mut nfi_buffer = [Weight(0); 5];
        let mut a_star_heap = [0u32; 5];
        let mut a_star_heap_pos = [0i32; 5];
        let mut f_score = [0u64; 5];
        let mut dlb = [false; 5];

        let workspace = Workspace {
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

        assert!(workspace.validate_for_graph(5));
        assert!(!workspace.validate_for_graph(6));
    }
}
