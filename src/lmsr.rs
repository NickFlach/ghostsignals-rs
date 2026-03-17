//! Core LMSR (Logarithmic Market Scoring Rule) mathematics
//!
//! This module implements the fundamental mathematical operations for LMSR markets,
//! with careful attention to numerical stability using the log-sum-exp trick.

use thiserror::Error;

/// Errors that can occur in LMSR calculations
#[derive(Error, Debug, Clone, PartialEq)]
pub enum LmsrError {
    #[error("Liquidity parameter b must be positive, got {0}")]
    InvalidLiquidity(f64),
    #[error("Empty quantities vector")]
    EmptyQuantities,
    #[error("Outcome index {outcome} out of bounds for {num_outcomes} outcomes")]
    InvalidOutcome { outcome: usize, num_outcomes: usize },
    #[error("Numerical overflow in calculation")]
    NumericalOverflow,
}

/// Calculate the LMSR cost function: C(q) = b * ln(Σ exp(qᵢ/b))
/// 
/// Uses the log-sum-exp trick for numerical stability to avoid overflow.
/// 
/// # Arguments
/// * `quantities` - Vector of quantities for each outcome
/// * `b` - Liquidity parameter (must be positive)
/// 
/// # Returns
/// The cost function value
/// 
/// # Errors
/// Returns error if liquidity is non-positive or quantities is empty
pub fn cost(quantities: &[f64], b: f64) -> Result<f64, LmsrError> {
    if b <= 0.0 {
        return Err(LmsrError::InvalidLiquidity(b));
    }
    if quantities.is_empty() {
        return Err(LmsrError::EmptyQuantities);
    }

    let normalized: Vec<f64> = quantities.iter().map(|&q| q / b).collect();
    let log_sum_exp_result = log_sum_exp(&normalized)?;
    
    Ok(b * log_sum_exp_result)
}

/// Calculate market prices: p(i) = exp(qᵢ/b) / Σ exp(qⱼ/b)
/// 
/// This implements the softmax function for converting quantities to probabilities.
/// Prices are guaranteed to sum to 1.0 (within floating point precision).
/// 
/// # Arguments
/// * `quantities` - Vector of quantities for each outcome
/// * `b` - Liquidity parameter (must be positive)
/// 
/// # Returns
/// Vector of probabilities/prices for each outcome
pub fn prices(quantities: &[f64], b: f64) -> Result<Vec<f64>, LmsrError> {
    if b <= 0.0 {
        return Err(LmsrError::InvalidLiquidity(b));
    }
    if quantities.is_empty() {
        return Err(LmsrError::EmptyQuantities);
    }

    let normalized: Vec<f64> = quantities.iter().map(|&q| q / b).collect();
    softmax(&normalized)
}

/// Calculate the cost of a trade: C(q_after) - C(q_before)
/// 
/// # Arguments
/// * `quantities` - Current quantities vector
/// * `b` - Liquidity parameter
/// * `outcome` - Index of outcome to trade
/// * `amount` - Amount to buy (positive) or sell (negative)
/// 
/// # Returns
/// The cost of the trade (positive = cost to trader, negative = payout to trader)
pub fn trade_cost(
    quantities: &[f64], 
    b: f64, 
    outcome: usize, 
    amount: f64
) -> Result<f64, LmsrError> {
    if outcome >= quantities.len() {
        return Err(LmsrError::InvalidOutcome {
            outcome,
            num_outcomes: quantities.len(),
        });
    }

    let cost_before = cost(quantities, b)?;
    
    let mut quantities_after = quantities.to_vec();
    quantities_after[outcome] += amount;
    
    let cost_after = cost(&quantities_after, b)?;
    
    Ok(cost_after - cost_before)
}

/// Log-sum-exp trick for numerical stability
/// 
/// Computes log(Σ exp(xᵢ)) without overflow by factoring out the maximum value.
fn log_sum_exp(values: &[f64]) -> Result<f64, LmsrError> {
    if values.is_empty() {
        return Err(LmsrError::EmptyQuantities);
    }

    // Find the maximum value to factor out
    let max_val = values.iter()
        .copied()
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap();

    // Handle edge cases
    if max_val.is_infinite() {
        if max_val.is_sign_positive() {
            return Err(LmsrError::NumericalOverflow);
        } else {
            return Ok(f64::NEG_INFINITY);
        }
    }

    // Compute log-sum-exp with max value factored out
    let sum: f64 = values.iter()
        .map(|&x| (x - max_val).exp())
        .sum();

    if sum <= 0.0 || !sum.is_finite() {
        return Err(LmsrError::NumericalOverflow);
    }

    Ok(max_val + sum.ln())
}

/// Softmax function with numerical stability
/// 
/// Computes exp(xᵢ) / Σ exp(xⱼ) for each i, ensuring the result sums to 1.0.
fn softmax(values: &[f64]) -> Result<Vec<f64>, LmsrError> {
    if values.is_empty() {
        return Err(LmsrError::EmptyQuantities);
    }

    let log_sum = log_sum_exp(values)?;
    
    let mut result: Vec<f64> = values.iter()
        .map(|&x| (x - log_sum).exp())
        .collect();

    // Normalize to ensure exact sum to 1.0
    let sum: f64 = result.iter().sum();
    if sum > 0.0 {
        for price in &mut result {
            *price /= sum;
        }
    }

    Ok(result)
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
    fn test_trade_cost_invalid_outcome() {
        let quantities = vec![0.0, 0.0];
        let result = trade_cost(&quantities, 100.0, 2, 10.0);
        assert!(matches!(result, Err(LmsrError::InvalidOutcome { outcome: 2, num_outcomes: 2 })));
    }

    #[test]
    fn test_log_sum_exp_stability() {
        // Test with large values that would overflow regular exp()
        let values = vec![700.0, 701.0, 699.0];
        let result = log_sum_exp(&values).unwrap();
        
        // Should not overflow and should be approximately 701 + ln(e^(-1) + 1 + e^(-2))
        assert!(result.is_finite());
        assert!(result > 701.0);
        assert!(result < 702.0);
    }

    #[test]
    fn test_log_sum_exp_small_values() {
        let values = vec![-700.0, -701.0, -699.0];
        let result = log_sum_exp(&values).unwrap();
        
        // Should handle large negative values without underflow
        assert!(result.is_finite());
        assert!(result > -700.0);
    }

    #[test]
    fn test_softmax_stability() {
        // Test with large values
        let values = vec![700.0, 701.0, 699.0];
        let result = softmax(&values).unwrap();
        
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