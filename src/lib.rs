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

use std::collections::HashMap;
use uuid::Uuid;
use serde::{Deserialize, Serialize};

mod lmsr;
mod market;
mod trading;
mod signals;

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

    /// Execute a trade
    pub fn trade(
        &mut self,
        market_id: MarketId,
        trader_id: TraderId,
        outcome: usize,
        amount: f64,
    ) -> Result<Trade, TradeError> {
        let market = self.markets.get_mut(&market_id)
            .ok_or(TradeError::MarketNotFound)?;

        // Ensure portfolio exists
        self.portfolios.entry(trader_id).or_insert_with(|| Portfolio::new(trader_id));

        let portfolio = self.portfolios.get_mut(&trader_id).unwrap();
        execute_trade(market, trader_id, outcome, amount, portfolio)
    }

    /// Get current market prices
    pub fn prices(&self, market_id: MarketId) -> Result<Vec<f64>, MarketError> {
        let market = self.markets.get(&market_id)
            .ok_or(MarketError::MarketNotFound)?;
        Ok(market.prices())
    }

    /// Resolve a market
    pub fn resolve(&mut self, market_id: MarketId, outcome: usize) -> Result<(), MarketError> {
        let market = self.markets.get_mut(&market_id)
            .ok_or(MarketError::MarketNotFound)?;
        market.resolve(outcome)
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
        
        // Add cash to portfolio first
        if !engine.portfolios.contains_key(&trader) {
            engine.portfolios.insert(trader, Portfolio::new(trader));
        }
        engine.portfolios.get_mut(&trader).unwrap().add_cash(1000.0);
        
        let trade = engine.trade(market_id, trader, 0, 10.0).unwrap();
        
        assert_eq!(trade.outcome, 0);
        assert_eq!(trade.amount, 10.0);
        assert!(trade.cost > 0.0);
        
        // Portfolio should be created
        assert!(engine.portfolio(trader).is_some());
    }
}