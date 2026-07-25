//! Fixed-point weight representation.

/// Represents a fixed-point weight scaled by $10^6$.
///
/// This type wraps a `u64` to avoid floating-point non-determinism and overhead
/// in bare-metal or `no_std` environments.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Weight(pub u64);

impl Weight {
    /// Creates a `Weight` from a floating-point value.
    ///
    /// This method scales the floating-point value by $10^6$ and rounds it to the
    /// nearest integer.
    ///
    /// # Examples
    ///
    /// ```
    /// # use dzul_core::Weight;
    /// let w = Weight::from_float(1.234_567);
    /// assert_eq!(w.0, 1_234_567);
    /// ```
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn from_float(val: f64) -> Self {
        let scaled = val * 1_000_000.0;
        let rounded = libm::round(scaled);
        Weight(rounded as u64)
    }

    /// Converts the `Weight` back to a floating-point value.
    ///
    /// This method divides the internal integer value by $10^6$ to recover the
    /// original floating-point representation.
    ///
    /// # Examples
    ///
    /// ```
    /// # use dzul_core::Weight;
    /// let w = Weight(1_234_567);
    /// assert_eq!(w.to_float(), 1.234_567);
    /// ```
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn to_float(self) -> f64 {
        self.0 as f64 / 1_000_000.0
    }

    /// Performs checked addition to prevent silent overflow.
    ///
    /// This method adds two `Weight` values and returns `None` if the operation
    /// overflows the underlying `u64` integer.
    ///
    /// # Examples
    ///
    /// ```
    /// # use dzul_core::Weight;
    /// let w1 = Weight(1_000_000);
    /// let w2 = Weight(2_000_000);
    /// assert_eq!(w1.checked_add(w2), Some(Weight(3_000_000)));
    ///
    /// let w_max = Weight(u64::MAX);
    /// assert_eq!(w_max.checked_add(Weight(1)), None);
    /// ```
    pub fn checked_add(self, other: Self) -> Option<Self> {
        self.0.checked_add(other.0).map(Weight)
    }

    /// Computes the standard TSPLIB `EUC_2D` distance between two 2D points.
    ///
    /// Applies nearest-integer rounding per edge as specified by TSPLIB:
    /// `dist = nint(sqrt(dx^2 + dy^2))`.
    ///
    /// This produces an exact integer weight (scaled by $10^6$) that matches
    /// Concorde TSPLIB benchmark optimal values without floating-point drift.
    ///
    /// # Examples
    ///
    /// ```
    /// # use dzul_core::Weight;
    /// let w = Weight::euc_2d(0.0, 0.0, 3.0, 4.0);
    /// assert_eq!(w.0, 5_000_000); // nint(5.0) = 5, scaled by 10^6
    /// ```
    // Anchor: FR-16
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn euc_2d(x1: f64, y1: f64, x2: f64, y2: f64) -> Self {
        let dx = x1 - x2;
        let dy = y1 - y2;
        let dist = libm::sqrt(dx * dx + dy * dy);
        let rounded = libm::floor(dist + 0.5);
        Weight((rounded as u64) * 1_000_000)
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    /// Tests the fixed-point weight representation and operations.
    // Anchor: FR-01
    #[test]
    #[allow(clippy::float_cmp)]
    fn test_fr_01_fixed_point() {
        let w1 = Weight::from_float(1.0);
        assert_eq!(w1.0, 1_000_000);
        assert_eq!(w1.to_float(), 1.0);

        let w2 = Weight::from_float(1.234_567);
        assert_eq!(w2.0, 1_234_567);
        assert_eq!(w2.to_float(), 1.234_567);

        let w3 = Weight::from_float(0.000_001);
        assert_eq!(w3.0, 1);
        assert_eq!(w3.to_float(), 0.000_001);

        let sum = w1.checked_add(w2);
        assert_eq!(sum, Some(Weight(2_234_567)));

        let overflow = Weight(u64::MAX).checked_add(Weight(1));
        assert_eq!(overflow, None);
    }

    /// Tests the TSPLIB `EUC_2D` distance mode with nearest-integer rounding.
    // Anchor: FR-16
    #[test]
    fn test_fr_16_euc_2d() {
        // 3-4-5 triangle: nint(5.0) = 5
        assert_eq!(Weight::euc_2d(0.0, 0.0, 3.0, 4.0), Weight(5_000_000));
        // Same point => 0
        assert_eq!(Weight::euc_2d(7.0, 7.0, 7.0, 7.0), Weight(0));
        // Rounding: sqrt(2) ≈ 1.414 => nint = 1
        assert_eq!(Weight::euc_2d(0.0, 0.0, 1.0, 1.0), Weight(1_000_000));
        // Rounding: sqrt(0.5^2+0.5^2) ≈ 0.707 => nint = 1
        assert_eq!(Weight::euc_2d(0.0, 0.0, 0.5, 0.5), Weight(1_000_000));
        // Negative coordinates: dx=3, dy=4 => 5
        assert_eq!(Weight::euc_2d(-1.0, -1.0, 2.0, 3.0), Weight(5_000_000));
    }
}
