//! Market management and lifecycle
//!
//! This module defines the Market struct and handles market creation,
//! state management, and resolution.

use crate::{lmsr, signals::MarketSignal, LmsrError};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

/// Errors that can occur in market operations
#[derive(Error, Debug, Clone, PartialEq)]
pub enum MarketError {
    #[error("Market not found")]
    MarketNotFound,
    #[error("Market is closed for trading")]
    MarketClosed,
    #[error("Market is already resolved")]
    MarketResolved,
    #[error("Invalid outcome index {outcome} for market with {num_outcomes} outcomes")]
    InvalidOutcome { outcome: usize, num_outcomes: usize },
    #[error("Must have at least 2 outcomes, got {0}")]
    InsufficientOutcomes(usize),
    #[error("Liquidity parameter must be positive, got {0}")]
    InvalidLiquidity(f64),
    #[error("Question cannot be empty")]
    EmptyQuestion,
    #[error("LMSR calculation error: {0}")]
    LmsrError(#[from] LmsrError),
}

/// Market state in its lifecycle
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MarketState {
    /// Market is open for trading
    Open,
    /// Market is closed but not yet resolved
    Closed,
    /// Market has been resolved with a winning outcome
    Resolved,
}

/// A prediction market
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Market {
    /// Unique market identifier
    id: Uuid,
    /// Market question
    question: String,
    /// Outcome labels
    outcomes: Vec<String>,
    /// Current quantities for each outcome
    quantities: Vec<f64>,
    /// Liquidity parameter (controls price sensitivity)
    liquidity: f64,
    /// Current market state
    state: MarketState,
    /// Market creation timestamp
    created_at: DateTime<Utc>,
    /// Resolved outcome index (if resolved)
    resolved_outcome: Option<usize>,
    /// Trading statistics
    trade_count: u64,
    total_volume: f64,
}

impl Market {
    /// Create a new market
    ///
    /// # Arguments
    /// * `question` - The market question
    /// * `outcomes` - Vector of outcome labels
    /// * `liquidity` - Liquidity parameter (must be positive)
    ///
    /// # Returns
    /// A new market with all quantities initialized to 0
    pub fn new(
        question: String,
        outcomes: Vec<String>,
        liquidity: f64,
    ) -> Result<Self, MarketError> {
        if question.trim().is_empty() {
            return Err(MarketError::EmptyQuestion);
        }
        if outcomes.len() < 2 {
            return Err(MarketError::InsufficientOutcomes(outcomes.len()));
        }
        if liquidity <= 0.0 {
            return Err(MarketError::InvalidLiquidity(liquidity));
        }

        let num_outcomes = outcomes.len();
        let quantities = vec![0.0; num_outcomes];

        Ok(Market {
            id: Uuid::new_v4(),
            question,
            outcomes,
            quantities,
            liquidity,
            state: MarketState::Open,
            created_at: Utc::now(),
            resolved_outcome: None,
            trade_count: 0,
            total_volume: 0.0,
        })
    }

    /// Get market ID
    pub fn id(&self) -> Uuid {
        self.id
    }

    /// Get market question
    pub fn question(&self) -> &str {
        &self.question
    }

    /// Get outcome labels
    pub fn outcomes(&self) -> &[String] {
        &self.outcomes
    }

    /// Get current quantities
    pub fn quantities(&self) -> &[f64] {
        &self.quantities
    }

    /// Get liquidity parameter
    pub fn liquidity(&self) -> f64 {
        self.liquidity
    }

    /// Get market state
    pub fn state(&self) -> MarketState {
        self.state
    }

    /// Get creation timestamp
    pub fn created_at(&self) -> DateTime<Utc> {
        self.created_at
    }

    /// Get resolved outcome (if any)
    pub fn resolved_outcome(&self) -> Option<usize> {
        self.resolved_outcome
    }

    /// Get trade statistics
    pub fn trade_stats(&self) -> (u64, f64) {
        (self.trade_count, self.total_volume)
    }

    /// Calculate current market prices
    #[must_use]
    pub fn prices(&self) -> Vec<f64> {
        lmsr::prices(&self.quantities, self.liquidity)
            .unwrap_or_else(|_| vec![1.0 / self.quantities.len() as f64; self.quantities.len()])
    }

    /// Calculate cost of a hypothetical trade
    pub fn trade_cost(&self, outcome: usize, amount: f64) -> Result<f64, MarketError> {
        if self.state != MarketState::Open {
            return Err(MarketError::MarketClosed);
        }
        if outcome >= self.outcomes.len() {
            return Err(MarketError::InvalidOutcome {
                outcome,
                num_outcomes: self.outcomes.len(),
            });
        }

        Ok(lmsr::trade_cost(&self.quantities, self.liquidity, outcome, amount)?)
    }

    /// Update market quantities after a trade
    ///
    /// This should only be called by the trading system
    pub(crate) fn apply_trade(
        &mut self,
        outcome: usize,
        amount: f64,
        cost: f64,
    ) -> Result<(), MarketError> {
        if self.state != MarketState::Open {
            return Err(MarketError::MarketClosed);
        }
        if outcome >= self.quantities.len() {
            return Err(MarketError::InvalidOutcome {
                outcome,
                num_outcomes: self.quantities.len(),
            });
        }

        self.quantities[outcome] += amount;
        self.trade_count += 1;
        self.total_volume += cost.abs();

        Ok(())
    }

    /// Close the market (stop trading but don't resolve yet)
    pub fn close(&mut self) -> Result<(), MarketError> {
        match self.state {
            MarketState::Open => {
                self.state = MarketState::Closed;
                Ok(())
            }
            MarketState::Closed => Ok(()), // Already closed
            MarketState::Resolved => Err(MarketError::MarketResolved),
        }
    }

    /// Resolve the market with a winning outcome
    pub fn resolve(&mut self, outcome: usize) -> Result<(), MarketError> {
        if outcome >= self.outcomes.len() {
            return Err(MarketError::InvalidOutcome {
                outcome,
                num_outcomes: self.outcomes.len(),
            });
        }

        match self.state {
            MarketState::Resolved => Err(MarketError::MarketResolved),
            _ => {
                self.state = MarketState::Resolved;
                self.resolved_outcome = Some(outcome);
                Ok(())
            }
        }
    }

    /// Generate a market signal snapshot
    #[must_use]
    pub fn signal(&self) -> MarketSignal {
        MarketSignal {
            market_id: self.id,
            prices: self.prices(),
            timestamp: Utc::now(),
            trade_count: self.trade_count,
            volume: self.total_volume,
        }
    }

    /// Calculate market entropy (uncertainty measure)
    ///
    /// Returns Shannon entropy of current price distribution.
    /// Higher values indicate more uncertainty.
    #[must_use]
    pub fn entropy(&self) -> f64 {
        crate::signals::entropy(&self.prices())
    }

    /// Check if market is tradeable
    pub fn is_tradeable(&self) -> bool {
        self.state == MarketState::Open
    }

    /// Get outcome name by index
    pub fn outcome_name(&self, index: usize) -> Option<&str> {
        self.outcomes.get(index).map(|s| s.as_str())
    }

    /// Find outcome index by name (case-insensitive)
    pub fn find_outcome(&self, name: &str) -> Option<usize> {
        self.outcomes
            .iter()
            .position(|outcome| outcome.eq_ignore_ascii_case(name))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_market_creation() {
        let market = Market::new(
            "Will it rain tomorrow?".to_string(),
            vec!["Yes".to_string(), "No".to_string()],
            100.0,
        )
        .unwrap();

        assert_eq!(market.question(), "Will it rain tomorrow?");
        assert_eq!(market.outcomes().len(), 2);
        assert_eq!(market.state(), MarketState::Open);
        assert!(market.resolved_outcome().is_none());
        assert_eq!(market.quantities(), &[0.0, 0.0]);
    }

    #[test]
    fn test_market_creation_validation() {
        // Empty question
        assert!(matches!(
            Market::new("".to_string(), vec!["A".to_string(), "B".to_string()], 100.0),
            Err(MarketError::EmptyQuestion)
        ));

        // Insufficient outcomes
        assert!(matches!(
            Market::new("Question?".to_string(), vec!["Only one".to_string()], 100.0),
            Err(MarketError::InsufficientOutcomes(1))
        ));

        // Invalid liquidity
        assert!(matches!(
            Market::new("Question?".to_string(), vec!["A".to_string(), "B".to_string()], 0.0),
            Err(MarketError::InvalidLiquidity(_))
        ));

        assert!(matches!(
            Market::new("Question?".to_string(), vec!["A".to_string(), "B".to_string()], -10.0),
            Err(MarketError::InvalidLiquidity(_))
        ));
    }

    #[test]
    fn test_initial_prices() {
        let market = Market::new(
            "Test".to_string(),
            vec!["A".to_string(), "B".to_string(), "C".to_string()],
            100.0,
        )
        .unwrap();

        let prices = market.prices();
        assert_eq!(prices.len(), 3);

        // All prices should be equal initially
        for price in &prices {
            assert_relative_eq!(*price, 1.0 / 3.0, epsilon = 1e-10);
        }

        // Prices should sum to 1
        let sum: f64 = prices.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-15);
    }

    #[test]
    fn test_trade_cost_calculation() {
        let market = Market::new(
            "Test".to_string(),
            vec!["A".to_string(), "B".to_string()],
            100.0,
        )
        .unwrap();

        let cost = market.trade_cost(0, 10.0).unwrap();
        assert!(cost > 0.0); // Should cost something to buy

        let cost_sell = market.trade_cost(0, -10.0).unwrap();
        assert!(cost_sell < 0.0); // Should get money back when selling
    }

    #[test]
    fn test_apply_trade() {
        let mut market = Market::new(
            "Test".to_string(),
            vec!["A".to_string(), "B".to_string()],
            100.0,
        )
        .unwrap();

        let initial_quantities = market.quantities().to_vec();
        market.apply_trade(0, 10.0, 5.0).unwrap();

        assert_eq!(market.quantities()[0], initial_quantities[0] + 10.0);
        assert_eq!(market.quantities()[1], initial_quantities[1]);

        let (trade_count, volume) = market.trade_stats();
        assert_eq!(trade_count, 1);
        assert_eq!(volume, 5.0);
    }

    #[test]
    fn test_market_lifecycle() {
        let mut market = Market::new(
            "Test".to_string(),
            vec!["A".to_string(), "B".to_string()],
            100.0,
        )
        .unwrap();

        // Initially open
        assert_eq!(market.state(), MarketState::Open);
        assert!(market.is_tradeable());

        // Can trade
        assert!(market.trade_cost(0, 10.0).is_ok());

        // Close market
        market.close().unwrap();
        assert_eq!(market.state(), MarketState::Closed);
        assert!(!market.is_tradeable());

        // Cannot trade when closed
        assert!(matches!(
            market.trade_cost(0, 10.0),
            Err(MarketError::MarketClosed)
        ));

        // Resolve market
        market.resolve(1).unwrap();
        assert_eq!(market.state(), MarketState::Resolved);
        assert_eq!(market.resolved_outcome(), Some(1));
        assert!(!market.is_tradeable());

        // Cannot resolve again
        assert!(matches!(
            market.resolve(0),
            Err(MarketError::MarketResolved)
        ));
    }

    #[test]
    fn test_outcome_helpers() {
        let market = Market::new(
            "Test".to_string(),
            vec!["Yes".to_string(), "No".to_string(), "Maybe".to_string()],
            100.0,
        )
        .unwrap();

        assert_eq!(market.outcome_name(0), Some("Yes"));
        assert_eq!(market.outcome_name(1), Some("No"));
        assert_eq!(market.outcome_name(2), Some("Maybe"));
        assert_eq!(market.outcome_name(3), None);

        assert_eq!(market.find_outcome("Yes"), Some(0));
        assert_eq!(market.find_outcome("yes"), Some(0)); // case insensitive
        assert_eq!(market.find_outcome("No"), Some(1));
        assert_eq!(market.find_outcome("Maybe"), Some(2));
        assert_eq!(market.find_outcome("Unknown"), None);
    }

    #[test]
    fn test_market_entropy() {
        let market = Market::new(
            "Test".to_string(),
            vec!["A".to_string(), "B".to_string()],
            100.0,
        )
        .unwrap();

        let entropy = market.entropy();
        
        // For equal probabilities (0.5, 0.5), entropy should be ln(2)
        assert_relative_eq!(entropy, 2.0_f64.ln(), epsilon = 1e-10);
    }

    #[test]
    fn test_market_signal() {
        let market = Market::new(
            "Test".to_string(),
            vec!["A".to_string(), "B".to_string()],
            100.0,
        )
        .unwrap();

        let signal = market.signal();
        assert_eq!(signal.market_id, market.id());
        assert_eq!(signal.prices.len(), 2);
        assert_eq!(signal.trade_count, 0);
        assert_eq!(signal.volume, 0.0);
    }

    #[test]
    fn test_invalid_outcome_operations() {
        let market = Market::new(
            "Test".to_string(),
            vec!["A".to_string(), "B".to_string()],
            100.0,
        )
        .unwrap();

        assert!(matches!(
            market.trade_cost(2, 10.0),
            Err(MarketError::InvalidOutcome { outcome: 2, num_outcomes: 2 })
        ));

        let mut market_mut = market;
        assert!(matches!(
            market_mut.resolve(2),
            Err(MarketError::InvalidOutcome { outcome: 2, num_outcomes: 2 })
        ));
    }

    #[test]
    fn test_multi_outcome_market() {
        let market = Market::new(
            "Who will win the race?".to_string(),
            vec![
                "Alice".to_string(),
                "Bob".to_string(),
                "Charlie".to_string(),
                "Diana".to_string(),
            ],
            200.0,
        )
        .unwrap();

        let prices = market.prices();
        assert_eq!(prices.len(), 4);

        // Each should have 25% probability initially
        for price in &prices {
            assert_relative_eq!(*price, 0.25, epsilon = 1e-10);
        }

        // Sum should be 1.0
        let sum: f64 = prices.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-15);
    }

    #[test]
    fn test_closed_market_trading() {
        let mut market = Market::new(
            "Test".to_string(),
            vec!["A".to_string(), "B".to_string()],
            100.0,
        )
        .unwrap();

        market.close().unwrap();

        assert!(matches!(
            market.apply_trade(0, 10.0, 5.0),
            Err(MarketError::MarketClosed)
        ));
    }
}