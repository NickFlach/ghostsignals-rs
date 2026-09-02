//! Market management and lifecycle
//!
//! This module defines the Market struct and handles market creation,
//! state management, and resolution.

use crate::{lmsr, signals::MarketSignal, LmsrError};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

/// Relative tolerance used to absorb floating-point dust when a sell brings
/// an outstanding quantity or a position back to zero.
pub(crate) const DUST_TOLERANCE: f64 = 1e-9;

/// Errors that can occur in market operations
#[derive(Error, Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum MarketError {
    #[error("Market not found")]
    MarketNotFound,
    #[error("Market is closed for trading")]
    MarketClosed,
    #[error("Market is already resolved")]
    MarketResolved,
    #[error("Market is not resolved")]
    MarketNotResolved,
    #[error("Invalid outcome index {outcome} for market with {num_outcomes} outcomes")]
    InvalidOutcome { outcome: usize, num_outcomes: usize },
    #[error("Must have at least 2 outcomes, got {0}")]
    InsufficientOutcomes(usize),
    #[error("Liquidity parameter must be positive and finite, got {0}")]
    InvalidLiquidity(f64),
    #[error("Question cannot be empty")]
    EmptyQuestion,
    #[error("Outcome label at index {0} cannot be empty")]
    EmptyOutcomeLabel(usize),
    #[error("Duplicate outcome label (case-insensitive): {0}")]
    DuplicateOutcomeLabel(String),
    /// A deserialized or externally constructed market violates an internal
    /// invariant (quantity vector length, negative quantity, resolution
    /// bookkeeping, ...).
    #[error("Invalid market state: {0}")]
    InvalidMarketState(String),
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
///
/// # Invariants
///
/// A `Market` value always satisfies [`Market::validate`]: the liquidity is
/// finite and positive, every outstanding quantity is finite and
/// non-negative, the quantity vector matches the outcome list, and the
/// current prices are computable. The invariants are enforced by
/// [`Market::new`], by trade application, and by deserialization (an invalid
/// serialized market is rejected instead of being loaded).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(try_from = "MarketRepr")]
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
    /// Net cash collected by the market maker (sum of trade costs, sells negative).
    #[serde(default)]
    pool: f64,
}

/// Wire representation of [`Market`]; deserialization goes through
/// [`Market::validate`] so an invalid serialized market is refused.
#[derive(Deserialize)]
struct MarketRepr {
    id: Uuid,
    question: String,
    outcomes: Vec<String>,
    quantities: Vec<f64>,
    liquidity: f64,
    state: MarketState,
    created_at: DateTime<Utc>,
    resolved_outcome: Option<usize>,
    trade_count: u64,
    total_volume: f64,
    #[serde(default)]
    pool: f64,
}

impl TryFrom<MarketRepr> for Market {
    type Error = MarketError;

    fn try_from(repr: MarketRepr) -> Result<Self, Self::Error> {
        let market = Market {
            id: repr.id,
            question: repr.question,
            outcomes: repr.outcomes,
            quantities: repr.quantities,
            liquidity: repr.liquidity,
            state: repr.state,
            created_at: repr.created_at,
            resolved_outcome: repr.resolved_outcome,
            trade_count: repr.trade_count,
            total_volume: repr.total_volume,
            pool: repr.pool,
        };
        market.validate()?;
        Ok(market)
    }
}

/// Check outcome labels: non-empty after trimming and unique ignoring ASCII
/// case (because [`Market::find_outcome`] is case-insensitive).
fn validate_outcomes(outcomes: &[String]) -> Result<(), MarketError> {
    if outcomes.len() < 2 {
        return Err(MarketError::InsufficientOutcomes(outcomes.len()));
    }
    for (i, label) in outcomes.iter().enumerate() {
        if label.trim().is_empty() {
            return Err(MarketError::EmptyOutcomeLabel(i));
        }
        if outcomes[..i].iter().any(|prev| prev.eq_ignore_ascii_case(label)) {
            return Err(MarketError::DuplicateOutcomeLabel(label.clone()));
        }
    }
    Ok(())
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
    ///
    /// # Errors
    /// * `EmptyQuestion` if the question is blank
    /// * `InsufficientOutcomes` for fewer than two outcomes
    /// * `EmptyOutcomeLabel` / `DuplicateOutcomeLabel` for blank or
    ///   (case-insensitively) repeated labels
    /// * `InvalidLiquidity` unless `liquidity` is finite and `> 0`
    pub fn new(
        question: String,
        outcomes: Vec<String>,
        liquidity: f64,
    ) -> Result<Self, MarketError> {
        if question.trim().is_empty() {
            return Err(MarketError::EmptyQuestion);
        }
        validate_outcomes(&outcomes)?;
        if !(liquidity.is_finite() && liquidity > 0.0) {
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
            pool: 0.0,
        })
    }

    /// Check every structural invariant of the market.
    ///
    /// Always `Ok` for a market produced by this crate; exposed so callers
    /// that persist markets can verify them explicitly.
    pub fn validate(&self) -> Result<(), MarketError> {
        if self.question.trim().is_empty() {
            return Err(MarketError::EmptyQuestion);
        }
        validate_outcomes(&self.outcomes)?;
        if !(self.liquidity.is_finite() && self.liquidity > 0.0) {
            return Err(MarketError::InvalidLiquidity(self.liquidity));
        }
        if self.quantities.len() != self.outcomes.len() {
            return Err(MarketError::InvalidMarketState(format!(
                "{} quantities for {} outcomes",
                self.quantities.len(),
                self.outcomes.len()
            )));
        }
        if let Some(&q) = self.quantities.iter().find(|q| !(q.is_finite() && **q >= 0.0)) {
            return Err(MarketError::InvalidMarketState(format!(
                "quantity {q} is negative or non-finite"
            )));
        }
        for (name, value) in [("total_volume", self.total_volume), ("pool", self.pool)] {
            if !value.is_finite() {
                return Err(MarketError::InvalidMarketState(format!("{name} is {value}")));
            }
        }
        match (self.state, self.resolved_outcome) {
            (MarketState::Resolved, Some(outcome)) if outcome < self.outcomes.len() => {}
            (MarketState::Resolved, other) => {
                return Err(MarketError::InvalidMarketState(format!(
                    "resolved market with resolved_outcome {other:?}"
                )));
            }
            (_, Some(outcome)) => {
                return Err(MarketError::InvalidMarketState(format!(
                    "unresolved market carries resolved_outcome {outcome}"
                )));
            }
            (_, None) => {}
        }
        // Prices must be computable (q / b must not overflow).
        lmsr::prices(&self.quantities, self.liquidity)?;
        Ok(())
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

    /// Get trade statistics: `(number of trades, gross volume)` where gross
    /// volume is the sum of `|cost|` over all trades.
    pub fn trade_stats(&self) -> (u64, f64) {
        (self.trade_count, self.total_volume)
    }

    /// Net cash collected by the market maker: the sum of every trade's cost
    /// (sells count negative).
    ///
    /// At resolution the market owes one unit per outstanding share of the
    /// winning outcome, so the maker's profit is `pool() - quantities()[w]`,
    /// which LMSR bounds below by `-liquidity() * ln(n)`.
    pub fn pool(&self) -> f64 {
        self.pool
    }

    /// Calculate current market prices
    ///
    /// Always finite, each in `[0, 1]`, summing to 1 within rounding. A market
    /// upholds [`Market::validate`], under which the LMSR price computation
    /// cannot fail; the uniform fallback is unreachable and only exists so
    /// this method never panics.
    #[must_use]
    pub fn prices(&self) -> Vec<f64> {
        match lmsr::prices(&self.quantities, self.liquidity) {
            Ok(prices) => prices,
            Err(_) => {
                debug_assert!(false, "market invariant violated: prices not computable");
                vec![1.0 / self.quantities.len() as f64; self.quantities.len()]
            }
        }
    }

    /// Calculate cost of a hypothetical trade
    ///
    /// `amount` must be finite; non-finite amounts are refused with an
    /// [`LmsrError::NonFinite`] wrapped in `MarketError::LmsrError`.
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
    /// This should only be called by the trading system, after the caller
    /// has verified the trader actually holds the shares being sold. The
    /// market is left unchanged if the resulting state would be invalid.
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
        if !amount.is_finite() || !cost.is_finite() {
            return Err(MarketError::LmsrError(LmsrError::NonFinite {
                what: "trade amount or cost",
                value: if amount.is_finite() { cost } else { amount },
            }));
        }

        let mut new_quantity = self.quantities[outcome] + amount;
        // Selling a whole position back can leave -1e-17 of dust; snap it to
        // zero so outstanding shares never go negative.
        if new_quantity < 0.0 {
            if new_quantity >= -DUST_TOLERANCE * amount.abs().max(1.0) {
                new_quantity = 0.0;
            } else {
                return Err(MarketError::InvalidMarketState(format!(
                    "selling {} shares of outcome {outcome} would leave {new_quantity} outstanding",
                    amount.abs()
                )));
            }
        }
        // The new state must still price; check before mutating.
        let mut after = self.quantities.clone();
        after[outcome] = new_quantity;
        lmsr::prices(&after, self.liquidity)?;

        self.quantities = after;
        self.trade_count += 1;
        self.total_volume += cost.abs();
        self.pool += cost;

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
    ///
    /// Resolution is idempotent: resolving an already-resolved market to the
    /// *same* outcome is a successful no-op, so a retried call is safe.
    /// Resolving it to a *different* outcome is refused with
    /// `MarketResolved` and leaves the original resolution in place.
    pub fn resolve(&mut self, outcome: usize) -> Result<(), MarketError> {
        if outcome >= self.outcomes.len() {
            return Err(MarketError::InvalidOutcome {
                outcome,
                num_outcomes: self.outcomes.len(),
            });
        }

        match self.state {
            MarketState::Resolved if self.resolved_outcome == Some(outcome) => Ok(()),
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

        assert!(matches!(
            Market::new("Question?".to_string(), vec!["A".to_string(), "B".to_string()], f64::NAN),
            Err(MarketError::InvalidLiquidity(_))
        ));

        assert!(matches!(
            Market::new("Question?".to_string(), vec!["A".to_string(), "".to_string()], 1.0),
            Err(MarketError::EmptyOutcomeLabel(1))
        ));

        assert!(matches!(
            Market::new("Question?".to_string(), vec!["A".to_string(), "a".to_string()], 1.0),
            Err(MarketError::DuplicateOutcomeLabel(_))
        ));
    }

    #[test]
    fn test_apply_trade_tracks_pool_and_snaps_dust() {
        let mut market = Market::new(
            "Test".to_string(),
            vec!["A".to_string(), "B".to_string()],
            100.0,
        )
        .unwrap();
        market.apply_trade(0, 0.3, 0.16).unwrap();
        market.apply_trade(0, -0.1, -0.05).unwrap();
        market.apply_trade(0, -0.1, -0.05).unwrap();
        // 0.3 - 0.1 - 0.1 - 0.1 = -2.8e-17 in f64: snapped to exactly 0.
        market.apply_trade(0, -0.1, -0.05).unwrap();
        assert_eq!(market.quantities()[0], 0.0);
        assert_relative_eq!(market.pool(), 0.01, epsilon = 1e-12);
        // A real over-sell is refused and leaves the market untouched.
        assert!(market.apply_trade(0, -1.0, -0.5).is_err());
        assert_eq!(market.quantities()[0], 0.0);
        assert_eq!(market.trade_stats().0, 4);
    }

    #[test]
    fn test_validate_rejects_inconsistent_state() {
        let mut market = Market::new(
            "Test".to_string(),
            vec!["A".to_string(), "B".to_string()],
            100.0,
        )
        .unwrap();
        assert!(market.validate().is_ok());
        market.resolved_outcome = Some(0);
        assert!(matches!(market.validate(), Err(MarketError::InvalidMarketState(_))));
        market.resolved_outcome = None;
        market.quantities = vec![0.0];
        assert!(matches!(market.validate(), Err(MarketError::InvalidMarketState(_))));
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