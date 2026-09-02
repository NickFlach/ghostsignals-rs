//! # ghostsignals
//!
//! Prediction markets as collective intelligence.
//!
//! Based on Hanson's Logarithmic Market Scoring Rule (LMSR),
//! ghostsignals turns agent opinions into calibrated probabilities.
//! When agents trade on what they believe, the market price converges
//! to the collective's true estimate — emergence from interference.
//!
//! Part of the ghostmagicOS ecosystem: dx/dt = f(x) - Iηx
//! The market IS the interference term.
//!
//! # Invariants
//!
//! The crate is the reference implementation for other GhostSignals ports,
//! so it is explicit about what it guarantees:
//!
//! * [`lmsr`]: `b` is finite and positive, quantities and amounts are finite;
//!   every result is finite or an error. Prices are in `[0, 1]` and sum to 1.
//! * [`Market`]: always satisfies [`Market::validate`], including after
//!   deserialization. Outstanding shares never go negative.
//! * [`Portfolio`]: cash is finite and never negative; a trader can never
//!   sell more than they hold; a whole position can always be closed.
//! * Money is conserved: the sum of all balances plus every market's
//!   [`Market::pool`] is constant across any sequence of trades, and the
//!   market maker's loss at settlement is bounded by `b · ln(n)`.
//!
//! The README examples below are compiled as doctests.
#![doc = include_str!("../README.md")]

use std::collections::HashMap;
use uuid::Uuid;
use serde::{Deserialize, Serialize};

pub mod lmsr;
pub mod market;
pub mod trading;
pub mod signals;

pub use lmsr::*;
pub use market::*;
pub use trading::*;
pub use signals::*;

/// High-level prediction market engine
///
/// The Engine manages multiple markets and trader portfolios,
/// providing a clean interface for market operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Engine {
    markets: HashMap<MarketId, Market>,
    portfolios: HashMap<TraderId, Portfolio>,
}

impl Engine {
    /// Create a new empty engine
    pub fn new() -> Self {
        Self {
            markets: HashMap::new(),
            portfolios: HashMap::new(),
        }
    }

    /// Create a new market
    ///
    /// Returns a unique market ID that can be used for subsequent operations.
    pub fn create_market(
        &mut self,
        question: String,
        outcomes: Vec<String>,
        liquidity: f64,
    ) -> Result<MarketId, MarketError> {
        let market = Market::new(question, outcomes, liquidity)?;
        let id = market.id();
        self.markets.insert(id, market);
        Ok(id)
    }

    /// Credit cash to a trader, creating the portfolio if needed
    ///
    /// # Errors
    /// `InvalidAmount` if `amount` is negative, NaN or infinite.
    pub fn deposit(&mut self, trader_id: TraderId, amount: f64) -> Result<(), TradeError> {
        self.portfolios
            .entry(trader_id)
            .or_insert_with(|| Portfolio::new(trader_id))
            .add_cash(amount)
    }

    /// Execute a trade
    ///
    /// `amount > 0` buys shares of `outcome`, `amount < 0` sells them. A
    /// portfolio is created on first use; fund it with [`Engine::deposit`]
    /// before buying. On error neither the market nor the portfolio changes.
    pub fn trade(
        &mut self,
        market_id: MarketId,
        trader_id: TraderId,
        outcome: usize,
        amount: f64,
    ) -> Result<Trade, TradeError> {
        let market = self.markets.get_mut(&market_id)
            .ok_or(TradeError::MarketNotFound)?;

        let portfolio = self
            .portfolios
            .entry(trader_id)
            .or_insert_with(|| Portfolio::new(trader_id));
        execute_trade(market, trader_id, outcome, amount, portfolio)
    }

    /// Get current market prices
    pub fn prices(&self, market_id: MarketId) -> Result<Vec<f64>, MarketError> {
        let market = self.markets.get(&market_id)
            .ok_or(MarketError::MarketNotFound)?;
        Ok(market.prices())
    }

    /// Resolve a market (idempotent for the same outcome, see [`Market::resolve`])
    pub fn resolve(&mut self, market_id: MarketId, outcome: usize) -> Result<(), MarketError> {
        let market = self.markets.get_mut(&market_id)
            .ok_or(MarketError::MarketNotFound)?;
        market.resolve(outcome)
    }

    /// Pay out a resolved market
    ///
    /// Every trader holding shares of the winning outcome is credited one
    /// unit per share; all positions in the market are then cleared. Returns
    /// `(trader, amount)` for each non-zero payout. Settling the same market
    /// again pays nothing, so the call is idempotent.
    ///
    /// # Errors
    /// `MarketNotFound`, or `MarketNotResolved` if the market has not been
    /// resolved yet.
    pub fn settle(&mut self, market_id: MarketId) -> Result<Vec<(TraderId, f64)>, MarketError> {
        let market = self.markets.get(&market_id)
            .ok_or(MarketError::MarketNotFound)?;
        let winner = match (market.state(), market.resolved_outcome()) {
            (MarketState::Resolved, Some(winner)) => winner,
            _ => return Err(MarketError::MarketNotResolved),
        };

        let mut payouts = Vec::new();
        for (trader_id, portfolio) in self.portfolios.iter_mut() {
            let paid = portfolio.settle_market(market_id, winner);
            if paid > 0.0 {
                payouts.push((*trader_id, paid));
            }
        }
        Ok(payouts)
    }

    /// Get market signal
    pub fn signal(&self, market_id: MarketId) -> Result<MarketSignal, MarketError> {
        let market = self.markets.get(&market_id)
            .ok_or(MarketError::MarketNotFound)?;
        Ok(market.signal())
    }

    /// Get a market by ID
    pub fn market(&self, market_id: MarketId) -> Option<&Market> {
        self.markets.get(&market_id)
    }

    /// Iterate over all markets
    pub fn markets(&self) -> impl Iterator<Item = &Market> {
        self.markets.values()
    }

    /// Get a trader's portfolio
    pub fn portfolio(&self, trader_id: TraderId) -> Option<&Portfolio> {
        self.portfolios.get(&trader_id)
    }
}

impl Default for Engine {
    fn default() -> Self {
        Self::new()
    }
}

/// Market identifier
pub type MarketId = Uuid;

/// Trader identifier
pub type TraderId = Uuid;

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_engine_creation() {
        let engine = Engine::new();
        assert!(engine.markets.is_empty());
        assert!(engine.portfolios.is_empty());
    }

    #[test]
    fn test_market_creation() {
        let mut engine = Engine::new();
        let market_id = engine
            .create_market(
                "Will it rain tomorrow?".to_string(),
                vec!["Yes".to_string(), "No".to_string()],
                100.0,
            )
            .unwrap();

        assert!(engine.market(market_id).is_some());
        assert_eq!(engine.markets().count(), 1);

        let prices = engine.prices(market_id).unwrap();
        assert_eq!(prices.len(), 2);
        assert_relative_eq!(prices[0] + prices[1], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_trading_flow() {
        let mut engine = Engine::new();
        let market_id = engine
            .create_market(
                "Test market".to_string(),
                vec!["A".to_string(), "B".to_string()],
                100.0,
            )
            .unwrap();

        let trader = Uuid::new_v4();

        // An unfunded trader cannot buy.
        assert!(matches!(
            engine.trade(market_id, trader, 0, 10.0),
            Err(TradeError::InsufficientFunds { .. })
        ));

        engine.deposit(trader, 1000.0).unwrap();
        let trade = engine.trade(market_id, trader, 0, 10.0).unwrap();

        assert_eq!(trade.outcome, 0);
        assert_eq!(trade.amount, 10.0);
        assert!(trade.cost > 0.0);

        // Portfolio should be created
        assert!(engine.portfolio(trader).is_some());
        assert_relative_eq!(
            engine.portfolio(trader).unwrap().cash(),
            1000.0 - trade.cost,
            epsilon = 1e-12
        );
    }

    #[test]
    fn test_unknown_market_and_trader() {
        let mut engine = Engine::new();
        let missing = Uuid::new_v4();
        assert!(matches!(engine.prices(missing), Err(MarketError::MarketNotFound)));
        assert!(matches!(engine.resolve(missing, 0), Err(MarketError::MarketNotFound)));
        assert!(matches!(engine.signal(missing), Err(MarketError::MarketNotFound)));
        assert!(matches!(
            engine.trade(missing, Uuid::new_v4(), 0, 1.0),
            Err(TradeError::MarketNotFound)
        ));
        assert!(engine.portfolio(missing).is_none());
    }
}
