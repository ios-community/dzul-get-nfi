//! Error types for the TSP solver.

use core::fmt;

/// Defines errors that can occur during TSP solver execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TspError {
    /// The graph structure is invalid or violates CSR invariants.
    InvalidGraph,
    /// No valid TSP tour could be found.
    NoTourFound,
    /// The backtracking limit was exceeded.
    BacktrackLimitExceeded,
    /// The provided workspace is too small for the graph.
    WorkspaceTooSmall,
    /// An arithmetic overflow occurred during weight calculations.
    ArithmeticOverflow,
}

impl fmt::Display for TspError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidGraph => write!(f, "invalid graph structure or CSR invariants violated"),
            Self::NoTourFound => write!(f, "no valid TSP tour could be found"),
            Self::BacktrackLimitExceeded => write!(f, "backtracking limit exceeded"),
            Self::WorkspaceTooSmall => write!(f, "workspace size is too small for the graph"),
            Self::ArithmeticOverflow => write!(f, "arithmetic overflow during weight calculation"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for TspError {}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    /// Tests the display formatting of [`TspError`].
    #[test]
    fn test_error_display() {
        assert_eq!(
            format!("{}", TspError::InvalidGraph),
            "invalid graph structure or CSR invariants violated"
        );
        assert_eq!(
            format!("{}", TspError::NoTourFound),
            "no valid TSP tour could be found"
        );
        assert_eq!(
            format!("{}", TspError::BacktrackLimitExceeded),
            "backtracking limit exceeded"
        );
        assert_eq!(
            format!("{}", TspError::WorkspaceTooSmall),
            "workspace size is too small for the graph"
        );
        assert_eq!(
            format!("{}", TspError::ArithmeticOverflow),
            "arithmetic overflow during weight calculation"
        );
    }
}
