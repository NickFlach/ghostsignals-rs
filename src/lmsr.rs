//! Core LMSR (Logarithmic Market Scoring Rule) mathematics
//!
//! This module implements the fundamental mathematical operations for LMSR markets,
//! with careful attention to numerical stability using the log-sum-exp trick.
//!
//! # Invariants
//!
//! Every public function in this module either returns a finite value or an
//! [`LmsrError`]. NaN and infinity never escape. Inputs are validated:
//!
//! * the liquidity parameter `b` must be finite and strictly positive;
//! * every quantity must be finite;
//! * a trade amount must be finite (zero is allowed here and costs nothing;
//!   the trading layer refuses it).

use thiserror::Error;

/// Errors that can occur in LMSR calculations
#[derive(Error, Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum LmsrError {
    /// The liquidity parameter was zero, negative, NaN or infinite.
    #[error("Liquidity parameter b must be positive and finite, got {0}")]
    InvalidLiquidity(f64),
    /// The quantities vector was empty.
    #[error("Empty quantities vector")]
    EmptyQuantities,
    /// The outcome index was out of range.
    #[error("Outcome index {outcome} out of bounds for {num_outcomes} outcomes")]
    InvalidOutcome { outcome: usize, num_outcomes: usize },
    /// A quantity or trade amount was NaN or infinite.
    #[error("Non-finite {what}: {value}")]
    NonFinite { what: &'static str, value: f64 },
    /// An intermediate value overflowed `f64` (for example `q / b` when `b`
    /// is tiny and `q` is huge).
    #[error("Numerical overflow in calculation")]
    NumericalOverflow,
}

fn check_liquidity(b: f64) -> Result<(), LmsrError> {
    if !(b.is_finite() && b > 0.0) {
        return Err(LmsrError::InvalidLiquidity(b));
    }
    Ok(())
}

fn check_quantities(quantities: &[f64]) -> Result<(), LmsrError> {
    if quantities.is_empty() {
        return Err(LmsrError::EmptyQuantities);
    }
    if let Some(&bad) = quantities.iter().find(|q| !q.is_finite()) {
        return Err(LmsrError::NonFinite {
            what: "quantity",
            value: bad,
        });
    }
    Ok(())
}

/// Divide every quantity by `b`, refusing the result if it overflowed.
fn normalize(quantities: &[f64], b: f64) -> Result<Vec<f64>, LmsrError> {
    let normalized: Vec<f64> = quantities.iter().map(|&q| q / b).collect();
    if normalized.iter().any(|x| !x.is_finite()) {
        return Err(LmsrError::NumericalOverflow);
    }
    Ok(normalized)
}

fn finite_or_overflow(value: f64) -> Result<f64, LmsrError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(LmsrError::NumericalOverflow)
    }
}

/// Calculate the LMSR cost function: C(q) = b * ln(Σ exp(qᵢ/b))
///
/// Uses the log-sum-exp trick for numerical stability to avoid overflow.
///
/// # Arguments
/// * `quantities` - Vector of quantities for each outcome (all finite)
/// * `b` - Liquidity parameter (finite and positive)
///
/// # Returns
/// The cost function value, always finite. It satisfies
/// `max(q) <= C(q) <= max(q) + b·ln(n)`.
///
/// # Errors
/// Returns an error if `b` is not finite and positive, if `quantities` is
/// empty or contains a non-finite value, or if `q / b` overflows.
pub fn cost(quantities: &[f64], b: f64) -> Result<f64, LmsrError> {
    check_liquidity(b)?;
    check_quantities(quantities)?;
    let normalized = normalize(quantities, b)?;
    finite_or_overflow(b * log_sum_exp(&normalized))
}

/// Calculate market prices: p(i) = exp(qᵢ/b) / Σ exp(qⱼ/b)
///
/// This implements the softmax function for converting quantities to probabilities.
///
/// # Guarantees
/// Every price is finite and in `[0, 1]`, and the prices sum to 1.0 within
/// floating-point precision. In exact arithmetic every price is strictly
/// positive; in `f64` a price can underflow to exactly `0.0` when an outcome
/// trails the leader by more than about `745·b`.
///
/// # Errors
/// Same conditions as [`cost`].
pub fn prices(quantities: &[f64], b: f64) -> Result<Vec<f64>, LmsrError> {
    check_liquidity(b)?;
    check_quantities(quantities)?;
    let normalized = normalize(quantities, b)?;
    Ok(softmax(&normalized))
}

/// Calculate the cost of a trade: C(q_after) - C(q_before)
///
/// # Arguments
/// * `quantities` - Current quantities vector
/// * `b` - Liquidity parameter
/// * `outcome` - Index of outcome to trade
/// * `amount` - Amount to buy (positive) or sell (negative); must be finite
///
/// # Returns
/// The cost of the trade (positive = cost to trader, negative = payout to trader).
/// Buying always costs at least 0, selling always pays at most `|amount|`.
///
/// # Numerical method
/// Rather than subtracting two evaluations of the cost function (which
/// cancels catastrophically for small trades), this uses the closed form
///
/// ```text
/// C(q + Δe_i) - C(q) = b · ln(1 + p_i · (exp(Δ/b) - 1))
/// ```
///
/// via `ln_1p`/`exp_m1`, and falls back to the direct log-sum-exp difference
/// only when the argument leaves the region where `ln_1p` is accurate (large
/// sells of a dominant outcome, or buys past `exp` overflow). Both branches
/// agree to within rounding where they overlap.
///
/// # Errors
/// Same conditions as [`cost`], plus `InvalidOutcome` for a bad index and
/// `NonFinite` for a NaN or infinite `amount`.
pub fn trade_cost(
    quantities: &[f64],
    b: f64,
    outcome: usize,
    amount: f64,
) -> Result<f64, LmsrError> {
    check_liquidity(b)?;
    check_quantities(quantities)?;
    if outcome >= quantities.len() {
        return Err(LmsrError::InvalidOutcome {
            outcome,
            num_outcomes: quantities.len(),
        });
    }
    if !amount.is_finite() {
        return Err(LmsrError::NonFinite {
            what: "trade amount",
            value: amount,
        });
    }
    if amount == 0.0 {
        return Ok(0.0);
    }

    let normalized = normalize(quantities, b)?;
    let d = amount / b;
    if !d.is_finite() {
        return Err(LmsrError::NumericalOverflow);
    }

    // Closed form: b·ln(1 + p_i·(e^d - 1)) with p_i = E_i / T computed from
    // max-shifted exponentials so T >= 1 and no term overflows.
    let max_val = max_finite(&normalized);
    let e_i = (normalized[outcome] - max_val).exp();
    let total: f64 = normalized.iter().map(|&x| (x - max_val).exp()).sum();
    let ratio = e_i * d.exp_m1() / total;

    // ln_1p is accurate for ratio > -0.5; beyond that (selling most of a
    // dominant position) or on overflow, the direct difference re-centres on
    // the new maximum and stays accurate.
    let result = if ratio.is_finite() && ratio > -0.5 {
        b * ratio.ln_1p()
    } else {
        let before_lse = log_sum_exp(&normalized);
        let mut after = normalized;
        after[outcome] += d;
        if !after[outcome].is_finite() {
            return Err(LmsrError::NumericalOverflow);
        }
        b * (log_sum_exp(&after) - before_lse)
    };

    finite_or_overflow(result)
}

/// Largest value of a non-empty slice of finite floats.
fn max_finite(values: &[f64]) -> f64 {
    values.iter().copied().fold(f64::NEG_INFINITY, f64::max)
}

/// Log-sum-exp trick for numerical stability
///
/// Computes log(Σ exp(xᵢ)) without overflow by factoring out the maximum value.
/// Input must be non-empty and finite (guaranteed by the public wrappers).
fn log_sum_exp(values: &[f64]) -> f64 {
    let max_val = max_finite(values);
    let sum: f64 = values.iter().map(|&x| (x - max_val).exp()).sum();
    max_val + sum.ln()
}

/// Softmax function with numerical stability
///
/// Computes exp(xᵢ) / Σ exp(xⱼ) for each i, ensuring the result sums to 1.0.
/// Input must be non-empty and finite (guaranteed by the public wrappers).
fn softmax(values: &[f64]) -> Vec<f64> {
    let max_val = max_finite(values);
    let mut result: Vec<f64> = values.iter().map(|&x| (x - max_val).exp()).collect();
    // The maximum term is exactly 1.0, so the sum is >= 1 and never zero.
    let sum: f64 = result.iter().sum();
    for price in &mut result {
        *price /= sum;
        // Guard against 1.0000000000000002 from the division rounding up.
        *price = price.clamp(0.0, 1.0);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_cost_basic() {
        let quantities = vec![0.0, 0.0];
        let b = 100.0;
        let c = cost(&quantities, b).unwrap();

        // Cost should be b * ln(2) for equal quantities
        assert_relative_eq!(c, b * 2.0_f64.ln(), epsilon = 1e-10);
    }

    #[test]
    fn test_cost_with_different_quantities() {
        let quantities = vec![10.0, 5.0];
        let b = 50.0;
        let c = cost(&quantities, b).unwrap();

        // Should be approximately b * ln(exp(10/50) + exp(5/50))
        let expected = b * ((10.0 / b).exp() + (5.0 / b).exp()).ln();
        assert_relative_eq!(c, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_cost_invalid_inputs() {
        assert!(matches!(cost(&[], 100.0), Err(LmsrError::EmptyQuantities)));
        assert!(matches!(cost(&[0.0, 0.0], 0.0), Err(LmsrError::InvalidLiquidity(_))));
        assert!(matches!(cost(&[0.0, 0.0], -10.0), Err(LmsrError::InvalidLiquidity(_))));
        assert!(matches!(cost(&[0.0, 0.0], f64::NAN), Err(LmsrError::InvalidLiquidity(_))));
        assert!(matches!(cost(&[0.0, 0.0], f64::INFINITY), Err(LmsrError::InvalidLiquidity(_))));
        assert!(matches!(cost(&[f64::NAN, 0.0], 1.0), Err(LmsrError::NonFinite { .. })));
        assert!(matches!(cost(&[f64::INFINITY, 0.0], 1.0), Err(LmsrError::NonFinite { .. })));
        // q / b overflows for tiny b and huge q.
        assert!(matches!(cost(&[1e300, 0.0], 1e-300), Err(LmsrError::NumericalOverflow)));
    }

    #[test]
    fn test_prices_sum_to_one() {
        let quantities = vec![0.0, 0.0, 0.0];
        let prices = prices(&quantities, 100.0).unwrap();

        let sum: f64 = prices.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-15);
    }

    #[test]
    fn test_prices_equal_quantities() {
        let quantities = vec![0.0, 0.0, 0.0];
        let prices = prices(&quantities, 100.0).unwrap();

        // All prices should be equal for equal quantities
        for price in &prices {
            assert_relative_eq!(*price, 1.0 / 3.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_prices_skewed_quantities() {
        let quantities = vec![500.0, 0.0];
        let prices = prices(&quantities, 50.0).unwrap();

        // First outcome should have much higher probability
        assert!(prices[0] > 0.9);
        assert!(prices[1] < 0.1);
        assert_relative_eq!(prices[0] + prices[1], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_prices_underflow_stays_in_unit_interval() {
        // Trailing by 100_000·b: exp underflows to exactly 0, leader is exactly 1.
        let p = prices(&[100_000.0, 0.0], 1.0).unwrap();
        assert_eq!(p, vec![1.0, 0.0]);
    }

    #[test]
    fn test_trade_cost_basic() {
        let quantities = vec![0.0, 0.0];
        let b = 100.0;
        let trade_cost_result = trade_cost(&quantities, b, 0, 10.0).unwrap();

        // Cost should be positive for buying
        assert!(trade_cost_result > 0.0);

        // Should match manual calculation
        let cost_before = cost(&quantities, b).unwrap();
        let mut quantities_after = quantities.clone();
        quantities_after[0] += 10.0;
        let cost_after = cost(&quantities_after, b).unwrap();
        let expected = cost_after - cost_before;

        assert_relative_eq!(trade_cost_result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_trade_cost_selling() {
        let quantities = vec![10.0, 0.0];
        let b = 100.0;
        let trade_cost_result = trade_cost(&quantities, b, 0, -5.0).unwrap();

        // Cost should be negative for selling (trader receives payout)
        assert!(trade_cost_result < 0.0);
    }

    #[test]
    fn test_trade_cost_zero_amount_is_free() {
        assert_eq!(trade_cost(&[3.0, 1.0], 7.0, 1, 0.0).unwrap(), 0.0);
    }

    #[test]
    fn test_trade_cost_invalid_outcome() {
        let quantities = vec![0.0, 0.0];
        let result = trade_cost(&quantities, 100.0, 2, 10.0);
        assert!(matches!(result, Err(LmsrError::InvalidOutcome { outcome: 2, num_outcomes: 2 })));
    }

    #[test]
    fn test_trade_cost_rejects_non_finite_amount() {
        for amount in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(matches!(
                trade_cost(&[0.0, 0.0], 100.0, 0, amount),
                Err(LmsrError::NonFinite { what: "trade amount", .. })
            ));
        }
    }

    #[test]
    fn test_trade_cost_fallback_branch_matches_closed_form() {
        // Selling nearly all of a dominant position takes the direct branch;
        // it must agree with a hand-computed C(after) - C(before).
        let q = [50.0, 0.0];
        let b = 10.0;
        let tc = trade_cost(&q, b, 0, -49.0).unwrap();
        let expected = cost(&[1.0, 0.0], b).unwrap() - cost(&q, b).unwrap();
        // Both sides are ~ -40 with rounding at 1e-14.
        assert_relative_eq!(tc, expected, max_relative = 1e-12);
    }

    #[test]
    fn test_log_sum_exp_stability() {
        // Test with large values that would overflow regular exp()
        let values = vec![700.0, 701.0, 699.0];
        let result = log_sum_exp(&values);

        // Should not overflow and should be approximately 701 + ln(e^(-1) + 1 + e^(-2))
        assert!(result.is_finite());
        assert!(result > 701.0);
        assert!(result < 702.0);
    }

    #[test]
    fn test_log_sum_exp_small_values() {
        let values = vec![-700.0, -701.0, -699.0];
        let result = log_sum_exp(&values);

        // Should handle large negative values without underflow
        assert!(result.is_finite());
        assert!(result > -700.0);
    }

    #[test]
    fn test_softmax_stability() {
        // Test with large values
        let values = vec![700.0, 701.0, 699.0];
        let result = softmax(&values);

        // Should sum to 1 and not contain infinities
        let sum: f64 = result.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-10);
        assert!(result.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_price_relationship() {
        // Test that higher quantities lead to higher prices
        let quantities = vec![10.0, 0.0, 5.0];
        let prices = prices(&quantities, 100.0).unwrap();

        assert!(prices[0] > prices[1]); // Higher quantity -> higher price
        assert!(prices[0] > prices[2]); // Higher quantity -> higher price
        assert!(prices[2] > prices[1]); // Higher quantity -> higher price
    }

    #[test]
    fn test_liquidity_parameter_effect() {
        let quantities = vec![10.0, 0.0];

        // Lower liquidity should mean more sensitive prices
        let prices_low_b = prices(&quantities, 10.0).unwrap();
        let prices_high_b = prices(&quantities, 1000.0).unwrap();

        // With low b, price difference should be more extreme
        let diff_low = prices_low_b[0] - prices_low_b[1];
        let diff_high = prices_high_b[0] - prices_high_b[1];

        assert!(diff_low > diff_high);
    }
}
