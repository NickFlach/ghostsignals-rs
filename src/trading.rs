//! Trade execution and portfolio management
//!
//! This module handles the execution of trades, tracking trader portfolios,
//! and calculating position values and payouts.
//!
//! # Invariants
//!
//! * A portfolio's cash balance is finite and never negative.
//! * Every stored position has finite, strictly positive `shares`; a position
//!   sold back to zero is removed.
//! * A sell can never exceed the shares held. Selling an amount within
//!   [`DUST_TOLERANCE`](crate::market) (relative) of the whole position
//!   closes it exactly, so floating-point dust never traps a trader in a
//!   position they cannot exit.
//! * [`execute_trade`] validates everything before mutating anything, so a
//!   failed trade leaves both the market and the portfolio untouched.

use crate::market::DUST_TOLERANCE;
use crate::{Market, MarketError, MarketId, TraderId};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;
use uuid::Uuid;

/// Errors that can occur during trading operations
#[derive(Error, Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum TradeError {
    #[error("Market not found")]
    MarketNotFound,
    #[error("Market is not open for trading")]
    MarketClosed,
    #[error("Insufficient funds: need {required}, have {available}")]
    InsufficientFunds { required: f64, available: f64 },
    #[error("Insufficient position: trying to sell {amount}, have {available}")]
    InsufficientPosition { amount: f64, available: f64 },
    #[error("Trade amount must be non-zero")]
    ZeroAmount,
    /// An amount (trade size, deposit, cost) was NaN, infinite, or had the
    /// wrong sign for the operation.
    #[error("Invalid amount: {0}")]
    InvalidAmount(f64),
    #[error("Market error: {0}")]
    MarketError(#[from] MarketError),
}

/// A completed trade
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    /// Unique trade identifier
    pub id: Uuid,
    /// Market this trade occurred in
    pub market_id: MarketId,
    /// Trader who made the trade
    pub trader_id: TraderId,
    /// Outcome that was traded
    pub outcome: usize,
    /// Amount traded (positive = buy, negative = sell)
    pub amount: f64,
    /// Cost of the trade (positive = cost to trader, negative = payout)
    pub cost: f64,
    /// Timestamp of the trade
    pub timestamp: DateTime<Utc>,
}

impl Trade {
    /// Create a new trade record
    pub fn new(
        market_id: MarketId,
        trader_id: TraderId,
        outcome: usize,
        amount: f64,
        cost: f64,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            market_id,
            trader_id,
            outcome,
            amount,
            cost,
            timestamp: Utc::now(),
        }
    }

    /// Check if this is a buy trade
    pub fn is_buy(&self) -> bool {
        self.amount > 0.0
    }

    /// Check if this is a sell trade
    pub fn is_sell(&self) -> bool {
        self.amount < 0.0
    }
}

/// Position in a single outcome of a market
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Position {
    /// Number of shares held
    pub shares: f64,
    /// Average cost basis
    pub avg_cost: f64,
    /// Total amount invested
    pub total_invested: f64,
}

impl Position {
    /// Create a new empty position
    pub fn new() -> Self {
        Self {
            shares: 0.0,
            avg_cost: 0.0,
            total_invested: 0.0,
        }
    }

    /// Whether the position is internally consistent: finite, non-negative
    /// shares and cost basis.
    fn is_valid(&self) -> bool {
        self.shares.is_finite()
            && self.shares >= 0.0
            && self.avg_cost.is_finite()
            && self.avg_cost >= 0.0
            && self.total_invested.is_finite()
            && self.total_invested >= 0.0
    }

    /// Update position after a trade
    ///
    /// `amount > 0` buys with `cost >= 0`; `amount < 0` sells with
    /// `cost <= 0`. A sell larger than the shares held (beyond dust
    /// tolerance) is refused with `InsufficientPosition`; a sell within
    /// tolerance of the whole position closes it exactly.
    ///
    /// # Errors
    /// `InvalidAmount` for non-finite inputs or a cost with the wrong sign;
    /// `InsufficientPosition` for an over-sell. The position is unchanged on
    /// error.
    pub fn apply_trade(&mut self, amount: f64, cost: f64) -> Result<(), TradeError> {
        if !amount.is_finite() {
            return Err(TradeError::InvalidAmount(amount));
        }
        if !cost.is_finite() {
            return Err(TradeError::InvalidAmount(cost));
        }

        if amount > 0.0 {
            if cost < 0.0 {
                return Err(TradeError::InvalidAmount(cost));
            }
            let new_total_invested = self.total_invested + cost;
            let new_shares = self.shares + amount;
            self.avg_cost = new_total_invested / new_shares;
            self.shares = new_shares;
            self.total_invested = new_total_invested;
        } else if amount < 0.0 {
            if cost > 0.0 {
                return Err(TradeError::InvalidAmount(cost));
            }
            let sell_amount = -amount;
            if sell_amount > self.shares * (1.0 + DUST_TOLERANCE) {
                return Err(TradeError::InsufficientPosition {
                    amount: sell_amount,
                    available: self.shares,
                });
            }
            if sell_amount >= self.shares * (1.0 - DUST_TOLERANCE) {
                // Closing the position: snap to exactly zero.
                self.shares = 0.0;
                self.avg_cost = 0.0;
                self.total_invested = 0.0;
            } else {
                let proportion_sold = sell_amount / self.shares;
                self.total_invested -= self.total_invested * proportion_sold;
                self.shares -= sell_amount;
            }
        }
        Ok(())
    }

    /// Calculate current value given market price
    pub fn value(&self, price: f64) -> f64 {
        self.shares * price
    }

    /// Calculate unrealized profit/loss
    pub fn unrealized_pnl(&self, price: f64) -> f64 {
        self.value(price) - self.total_invested
    }

    /// Calculate profit/loss if position were closed at given price
    pub fn pnl_if_closed(&self, price: f64) -> f64 {
        self.unrealized_pnl(price)
    }
}

impl Default for Position {
    fn default() -> Self {
        Self::new()
    }
}

/// A trader's portfolio across all markets
///
/// Cash is only reachable through [`Portfolio::cash`] / [`Portfolio::add_cash`]
/// and trade execution, which keeps it finite and non-negative. Deserializing
/// a portfolio that violates that (or holds a non-positive position) fails.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(try_from = "PortfolioRepr")]
pub struct Portfolio {
    /// Trader identifier
    pub trader_id: TraderId,
    /// Positions by market and outcome
    positions: HashMap<MarketId, HashMap<usize, Position>>,
    /// Trade history
    trades: Vec<Trade>,
    /// Available cash balance
    cash: f64,
}

/// Wire representation of [`Portfolio`]; validated on deserialization.
#[derive(Deserialize)]
struct PortfolioRepr {
    trader_id: TraderId,
    positions: HashMap<MarketId, HashMap<usize, Position>>,
    trades: Vec<Trade>,
    cash: f64,
}

impl TryFrom<PortfolioRepr> for Portfolio {
    type Error = TradeError;

    fn try_from(repr: PortfolioRepr) -> Result<Self, Self::Error> {
        if !(repr.cash.is_finite() && repr.cash >= 0.0) {
            return Err(TradeError::InvalidAmount(repr.cash));
        }
        for position in repr.positions.values().flat_map(HashMap::values) {
            if !position.is_valid() || position.shares == 0.0 {
                return Err(TradeError::InvalidAmount(position.shares));
            }
        }
        if let Some(bad) = repr
            .trades
            .iter()
            .find(|t| !(t.amount.is_finite() && t.cost.is_finite()))
        {
            return Err(TradeError::InvalidAmount(if bad.amount.is_finite() {
                bad.cost
            } else {
                bad.amount
            }));
        }
        Ok(Portfolio {
            trader_id: repr.trader_id,
            positions: repr.positions,
            trades: repr.trades,
            cash: repr.cash,
        })
    }
}

impl Portfolio {
    /// Create a new portfolio for a trader
    pub fn new(trader_id: TraderId) -> Self {
        Self {
            trader_id,
            positions: HashMap::new(),
            trades: Vec::new(),
            cash: 0.0,
        }
    }

    /// Available cash balance (finite, never negative)
    pub fn cash(&self) -> f64 {
        self.cash
    }

    /// Add cash to the portfolio
    ///
    /// # Errors
    /// `InvalidAmount` if `amount` is negative, NaN or infinite, or if the
    /// resulting balance would overflow `f64`.
    pub fn add_cash(&mut self, amount: f64) -> Result<(), TradeError> {
        if !(amount.is_finite() && amount >= 0.0) {
            return Err(TradeError::InvalidAmount(amount));
        }
        let new_cash = self.cash + amount;
        if !new_cash.is_finite() {
            return Err(TradeError::InvalidAmount(amount));
        }
        self.cash = new_cash;
        Ok(())
    }

    /// Get position in a specific outcome
    pub fn position(&self, market_id: MarketId, outcome: usize) -> Option<&Position> {
        self.positions.get(&market_id)?.get(&outcome)
    }

    /// Get all positions in a market
    pub fn market_positions(&self, market_id: MarketId) -> Option<&HashMap<usize, Position>> {
        self.positions.get(&market_id)
    }

    /// Get all markets this trader has positions in
    pub fn markets(&self) -> impl Iterator<Item = MarketId> + '_ {
        self.positions.keys().copied()
    }

    /// Verify a trade can be applied without changing anything.
    pub(crate) fn check_trade(&self, trade: &Trade) -> Result<(), TradeError> {
        if !(trade.amount.is_finite() && trade.cost.is_finite()) {
            return Err(TradeError::InvalidAmount(if trade.amount.is_finite() {
                trade.cost
            } else {
                trade.amount
            }));
        }
        if trade.amount == 0.0 {
            return Err(TradeError::ZeroAmount);
        }

        if trade.amount > 0.0 {
            if trade.cost < 0.0 {
                return Err(TradeError::InvalidAmount(trade.cost));
            }
            if trade.cost > self.cash {
                return Err(TradeError::InsufficientFunds {
                    required: trade.cost,
                    available: self.cash,
                });
            }
        } else {
            if trade.cost > 0.0 {
                return Err(TradeError::InvalidAmount(trade.cost));
            }
            let current_shares = self.shares(trade.market_id, trade.outcome);
            let sell_amount = -trade.amount;
            if sell_amount > current_shares * (1.0 + DUST_TOLERANCE) {
                return Err(TradeError::InsufficientPosition {
                    amount: sell_amount,
                    available: current_shares,
                });
            }
        }
        Ok(())
    }

    /// Apply a trade that has already passed [`Portfolio::check_trade`].
    pub(crate) fn commit_trade(&mut self, trade: Trade) {
        // Cash: a buy subtracts cost <= cash (never negative), a sell adds |cost|.
        self.cash = (self.cash - trade.cost).max(0.0);

        let market_positions = self.positions.entry(trade.market_id).or_default();
        let position = market_positions.entry(trade.outcome).or_default();
        // Cannot fail: check_trade enforced the same bounds.
        let _ = position.apply_trade(trade.amount, trade.cost);

        // Remove empty positions to keep things clean
        if position.shares <= 0.0 {
            market_positions.remove(&trade.outcome);
            if market_positions.is_empty() {
                self.positions.remove(&trade.market_id);
            }
        }

        self.trades.push(trade);
    }

    /// Pay out a resolved market: credit one unit per share of the winning
    /// outcome and drop every position in that market. Returns the amount
    /// credited (0 when the trader held nothing in the market).
    pub(crate) fn settle_market(&mut self, market_id: MarketId, winning_outcome: usize) -> f64 {
        let Some(market_positions) = self.positions.remove(&market_id) else {
            return 0.0;
        };
        let payout = market_positions
            .get(&winning_outcome)
            .map(|position| position.shares)
            .unwrap_or(0.0);
        self.cash += payout;
        payout
    }

    /// Calculate total portfolio value
    pub fn total_value(&self, markets: &HashMap<MarketId, &Market>) -> f64 {
        let mut total = self.cash;

        for (market_id, market_positions) in &self.positions {
            if let Some(market) = markets.get(market_id) {
                let prices = market.prices();
                for (outcome, position) in market_positions {
                    if let Some(&price) = prices.get(*outcome) {
                        total += position.value(price);
                    }
                }
            }
        }

        total
    }

    /// Calculate portfolio value in a specific market
    pub fn market_value(&self, market: &Market) -> f64 {
        let market_id = market.id();
        let Some(market_positions) = self.positions.get(&market_id) else {
            return 0.0;
        };

        let prices = market.prices();
        let mut total = 0.0;

        for (outcome, position) in market_positions {
            if let Some(&price) = prices.get(*outcome) {
                total += position.value(price);
            }
        }

        total
    }

    /// Calculate payout if a market resolves to a specific outcome
    pub fn market_payout(&self, market: &Market, resolved_outcome: usize) -> f64 {
        let market_id = market.id();
        let Some(market_positions) = self.positions.get(&market_id) else {
            return 0.0;
        };

        // Only the resolved outcome pays out $1 per share
        market_positions
            .get(&resolved_outcome)
            .map(|position| position.shares)
            .unwrap_or(0.0)
    }

    /// Get trade history
    pub fn trades(&self) -> &[Trade] {
        &self.trades
    }

    /// Get trade history for a specific market
    pub fn market_trades(&self, market_id: MarketId) -> Vec<&Trade> {
        self.trades
            .iter()
            .filter(|trade| trade.market_id == market_id)
            .collect()
    }

    /// Calculate total profit/loss
    pub fn total_pnl(&self, markets: &HashMap<MarketId, &Market>) -> f64 {
        let current_value = self.total_value(markets);
        let total_invested: f64 = self.trades.iter().map(|t| t.cost).sum();
        current_value - total_invested
    }

    /// Check if trader can afford a trade
    pub fn can_afford(&self, cost: f64) -> bool {
        cost <= self.cash
    }

    /// Get number of shares in an outcome
    pub fn shares(&self, market_id: MarketId, outcome: usize) -> f64 {
        self.position(market_id, outcome)
            .map(|p| p.shares)
            .unwrap_or(0.0)
    }
}

/// Execute a trade
///
/// Validates the amount, prices the trade, checks funds or position, and
/// only then updates the market and the portfolio. On any error neither is
/// modified.
///
/// A sell whose size is within [`DUST_TOLERANCE`](crate::market) (relative)
/// of the shares held is treated as selling the whole position, so the
/// returned [`Trade::amount`] may differ from the requested amount by that
/// much.
///
/// # Errors
/// * `ZeroAmount` / `InvalidAmount` for `0`, NaN or infinite amounts
/// * `MarketClosed` if the market is not open
/// * `InsufficientFunds` / `InsufficientPosition`
/// * `MarketError` for anything the market refuses (bad outcome, overflow)
pub fn execute_trade(
    market: &mut Market,
    trader_id: TraderId,
    outcome: usize,
    amount: f64,
    portfolio: &mut Portfolio,
) -> Result<Trade, TradeError> {
    if !amount.is_finite() {
        return Err(TradeError::InvalidAmount(amount));
    }
    if amount == 0.0 {
        return Err(TradeError::ZeroAmount);
    }
    if !market.is_tradeable() {
        return Err(TradeError::MarketClosed);
    }

    // Snap a whole-position sell to the exact holding so dust cannot block it.
    let mut amount = amount;
    if amount < 0.0 {
        let held = portfolio.shares(market.id(), outcome);
        let sell_amount = -amount;
        if sell_amount > held * (1.0 + DUST_TOLERANCE) {
            return Err(TradeError::InsufficientPosition {
                amount: sell_amount,
                available: held,
            });
        }
        if sell_amount >= held * (1.0 - DUST_TOLERANCE) {
            amount = -held;
        }
    }

    // Calculate trade cost
    let cost = match market.trade_cost(outcome, amount) {
        Ok(cost) => cost,
        Err(MarketError::MarketClosed) => return Err(TradeError::MarketClosed),
        Err(e) => return Err(e.into()),
    };

    // Create trade record and validate against the portfolio (no mutation yet)
    let trade = Trade::new(market.id(), trader_id, outcome, amount, cost);
    portfolio.check_trade(&trade)?;

    // Apply to market (checks its own invariants before mutating), then commit
    // to the portfolio, which cannot fail after check_trade.
    market.apply_trade(outcome, amount, cost)?;
    portfolio.commit_trade(trade.clone());

    Ok(trade)
}

/// Calculate current portfolio value
pub fn portfolio_value(portfolio: &Portfolio, market: &Market) -> f64 {
    portfolio.market_value(market)
}

/// Calculate payout if market resolves to current prices
pub fn payout(portfolio: &Portfolio, market: &Market) -> f64 {
    // This function simulates expected payout based on current prices
    let market_id = market.id();
    let Some(market_positions) = portfolio.positions.get(&market_id) else {
        return 0.0;
    };

    let prices = market.prices();
    let mut expected_payout = 0.0;

    for (outcome, position) in market_positions {
        if let Some(&price) = prices.get(*outcome) {
            // Expected payout is probability * shares * $1
            expected_payout += price * position.shares;
        }
    }

    expected_payout
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Market;
    use approx::assert_relative_eq;

    #[test]
    fn test_position_creation() {
        let pos = Position::new();
        assert_eq!(pos.shares, 0.0);
        assert_eq!(pos.avg_cost, 0.0);
        assert_eq!(pos.total_invested, 0.0);
    }

    #[test]
    fn test_position_buy_trade() {
        let mut pos = Position::new();
        pos.apply_trade(10.0, 5.0).unwrap();

        assert_eq!(pos.shares, 10.0);
        assert_eq!(pos.avg_cost, 0.5);
        assert_eq!(pos.total_invested, 5.0);
    }

    #[test]
    fn test_position_multiple_buys() {
        let mut pos = Position::new();
        pos.apply_trade(10.0, 5.0).unwrap(); // $0.50 per share
        pos.apply_trade(5.0, 3.0).unwrap(); // $0.60 per share

        assert_eq!(pos.shares, 15.0);
        assert_relative_eq!(pos.avg_cost, 8.0 / 15.0, epsilon = 1e-10);
        assert_eq!(pos.total_invested, 8.0);
    }

    #[test]
    fn test_position_sell_trade() {
        let mut pos = Position::new();
        pos.apply_trade(10.0, 5.0).unwrap();
        pos.apply_trade(-3.0, -2.0).unwrap(); // Sell 3 shares

        assert_eq!(pos.shares, 7.0);
        assert_relative_eq!(pos.total_invested, 3.5, epsilon = 1e-10); // 70% of original
    }

    #[test]
    fn test_position_over_sell_is_refused() {
        let mut pos = Position::new();
        pos.apply_trade(10.0, 5.0).unwrap();
        let err = pos.apply_trade(-20.0, -8.0).unwrap_err();
        assert!(matches!(err, TradeError::InsufficientPosition { .. }));
        assert_eq!(pos.shares, 10.0);
        assert_eq!(pos.total_invested, 5.0);
    }

    #[test]
    fn test_position_rejects_non_finite_and_wrong_sign() {
        let mut pos = Position::new();
        assert!(matches!(pos.apply_trade(f64::NAN, 1.0), Err(TradeError::InvalidAmount(_))));
        assert!(matches!(pos.apply_trade(1.0, f64::INFINITY), Err(TradeError::InvalidAmount(_))));
        assert!(matches!(pos.apply_trade(1.0, -1.0), Err(TradeError::InvalidAmount(_))));
        pos.apply_trade(1.0, 0.5).unwrap();
        assert!(matches!(pos.apply_trade(-1.0, 0.5), Err(TradeError::InvalidAmount(_))));
        assert_eq!(pos.shares, 1.0);
    }

    #[test]
    fn test_position_dust_close() {
        let mut pos = Position::new();
        pos.apply_trade(0.3, 0.15).unwrap();
        pos.apply_trade(-0.1, -0.05).unwrap();
        pos.apply_trade(-0.1, -0.05).unwrap();
        // 0.09999999999999998 shares left; selling 0.1 closes it exactly.
        pos.apply_trade(-0.1, -0.05).unwrap();
        assert_eq!(pos.shares, 0.0);
        assert_eq!(pos.total_invested, 0.0);
        assert_eq!(pos.avg_cost, 0.0);
    }

    #[test]
    fn test_position_value() {
        let mut pos = Position::new();
        pos.apply_trade(10.0, 5.0).unwrap();

        assert_eq!(pos.value(0.6), 6.0); // 10 shares * $0.60
        assert_eq!(pos.unrealized_pnl(0.6), 1.0); // $6 - $5
    }

    #[test]
    fn test_portfolio_creation() {
        let trader_id = uuid::Uuid::new_v4();
        let portfolio = Portfolio::new(trader_id);

        assert_eq!(portfolio.trader_id, trader_id);
        assert_eq!(portfolio.cash(), 0.0);
        assert!(portfolio.trades().is_empty());
    }

    #[test]
    fn test_portfolio_cash_management() {
        let trader_id = uuid::Uuid::new_v4();
        let mut portfolio = Portfolio::new(trader_id);

        portfolio.add_cash(1000.0).unwrap();
        assert_eq!(portfolio.cash(), 1000.0);

        assert!(portfolio.can_afford(500.0));
        assert!(!portfolio.can_afford(1500.0));

        for bad in [-1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(matches!(portfolio.add_cash(bad), Err(TradeError::InvalidAmount(_))));
        }
        assert_eq!(portfolio.cash(), 1000.0);
        // Adding zero is a harmless no-op.
        portfolio.add_cash(0.0).unwrap();
        assert_eq!(portfolio.cash(), 1000.0);
        // Overflowing the balance is refused.
        portfolio.add_cash(f64::MAX).unwrap();
        assert!(matches!(portfolio.add_cash(f64::MAX), Err(TradeError::InvalidAmount(_))));
    }

    #[test]
    fn test_execute_trade_success() {
        let trader_id = uuid::Uuid::new_v4();
        let mut portfolio = Portfolio::new(trader_id);
        portfolio.add_cash(1000.0).unwrap();

        let mut market = Market::new(
            "Test".to_string(),
            vec!["A".to_string(), "B".to_string()],
            100.0,
        )
        .unwrap();

        let trade = execute_trade(&mut market, trader_id, 0, 10.0, &mut portfolio).unwrap();

        assert_eq!(trade.trader_id, trader_id);
        assert_eq!(trade.outcome, 0);
        assert_eq!(trade.amount, 10.0);
        assert!(trade.cost > 0.0);

        // Check portfolio was updated
        assert!(portfolio.cash() < 1000.0);
        assert_eq!(portfolio.shares(market.id(), 0), 10.0);
        assert_eq!(market.pool(), trade.cost);
    }

    #[test]
    fn test_execute_trade_insufficient_funds() {
        let trader_id = uuid::Uuid::new_v4();
        let mut portfolio = Portfolio::new(trader_id);
        portfolio.add_cash(1.0).unwrap(); // Very little cash

        let mut market = Market::new(
            "Test".to_string(),
            vec!["A".to_string(), "B".to_string()],
            100.0,
        )
        .unwrap();

        let result = execute_trade(&mut market, trader_id, 0, 100.0, &mut portfolio);
        assert!(matches!(result, Err(TradeError::InsufficientFunds { .. })));
        // Nothing changed on either side.
        assert_eq!(portfolio.cash(), 1.0);
        assert_eq!(market.trade_stats(), (0, 0.0));
        assert_eq!(market.quantities(), &[0.0, 0.0]);
    }

    #[test]
    fn test_execute_sell_insufficient_position() {
        let trader_id = uuid::Uuid::new_v4();
        let mut portfolio = Portfolio::new(trader_id);
        portfolio.add_cash(1000.0).unwrap();

        let mut market = Market::new(
            "Test".to_string(),
            vec!["A".to_string(), "B".to_string()],
            100.0,
        )
        .unwrap();

        // Try to sell without owning any shares
        let result = execute_trade(&mut market, trader_id, 0, -10.0, &mut portfolio);
        assert!(matches!(result, Err(TradeError::InsufficientPosition { .. })));
        assert_eq!(market.quantities(), &[0.0, 0.0]);
    }

    #[test]
    fn test_buy_then_sell() {
        let trader_id = uuid::Uuid::new_v4();
        let mut portfolio = Portfolio::new(trader_id);
        portfolio.add_cash(1000.0).unwrap();

        let mut market = Market::new(
            "Test".to_string(),
            vec!["A".to_string(), "B".to_string()],
            100.0,
        )
        .unwrap();

        // Buy first
        let _buy_trade = execute_trade(&mut market, trader_id, 0, 10.0, &mut portfolio).unwrap();
        let cash_after_buy = portfolio.cash();

        // Then sell some
        let sell_trade = execute_trade(&mut market, trader_id, 0, -5.0, &mut portfolio).unwrap();

        assert!(sell_trade.cost < 0.0); // Should get money back
        assert!(portfolio.cash() > cash_after_buy); // Cash should increase
        assert_eq!(portfolio.shares(market.id(), 0), 5.0); // 5 shares left
    }

    #[test]
    fn test_portfolio_value_calculation() {
        let trader_id = uuid::Uuid::new_v4();
        let mut portfolio = Portfolio::new(trader_id);
        portfolio.add_cash(1000.0).unwrap();

        let mut market = Market::new(
            "Test".to_string(),
            vec!["A".to_string(), "B".to_string()],
            100.0,
        )
        .unwrap();

        // Execute some trades
        execute_trade(&mut market, trader_id, 0, 10.0, &mut portfolio).unwrap();
        execute_trade(&mut market, trader_id, 1, 5.0, &mut portfolio).unwrap();

        let market_value = portfolio_value(&portfolio, &market);
        assert!(market_value > 0.0);

        let expected_payout = payout(&portfolio, &market);
        assert_relative_eq!(market_value, expected_payout, epsilon = 1e-10);
    }

    #[test]
    fn test_market_payout_on_resolution() {
        let trader_id = uuid::Uuid::new_v4();
        let mut portfolio = Portfolio::new(trader_id);
        portfolio.add_cash(1000.0).unwrap();

        let mut market = Market::new(
            "Test".to_string(),
            vec!["A".to_string(), "B".to_string()],
            100.0,
        )
        .unwrap();

        // Buy some shares of outcome 0
        execute_trade(&mut market, trader_id, 0, 10.0, &mut portfolio).unwrap();
        execute_trade(&mut market, trader_id, 1, 5.0, &mut portfolio).unwrap();

        // Outcome 0 wins
        let payout_0 = portfolio.market_payout(&market, 0);
        assert_eq!(payout_0, 10.0); // 10 shares * $1

        // Outcome 1 wins
        let payout_1 = portfolio.market_payout(&market, 1);
        assert_eq!(payout_1, 5.0); // 5 shares * $1

        // Settling credits the winning shares and clears the market's positions.
        let cash_before = portfolio.cash();
        assert_eq!(portfolio.settle_market(market.id(), 1), 5.0);
        assert_eq!(portfolio.cash(), cash_before + 5.0);
        assert!(portfolio.market_positions(market.id()).is_none());
        assert_eq!(portfolio.settle_market(market.id(), 1), 0.0);
    }

    #[test]
    fn test_trade_history() {
        let trader_id = uuid::Uuid::new_v4();
        let mut portfolio = Portfolio::new(trader_id);
        portfolio.add_cash(1000.0).unwrap();

        let mut market = Market::new(
            "Test".to_string(),
            vec!["A".to_string(), "B".to_string()],
            100.0,
        )
        .unwrap();

        execute_trade(&mut market, trader_id, 0, 10.0, &mut portfolio).unwrap();
        execute_trade(&mut market, trader_id, 1, 5.0, &mut portfolio).unwrap();

        let trades = portfolio.trades();
        assert_eq!(trades.len(), 2);
        assert!(trades[0].is_buy());
        assert!(trades[1].is_buy());

        let market_trades = portfolio.market_trades(market.id());
        assert_eq!(market_trades.len(), 2);
    }

    #[test]
    fn test_zero_amount_trade() {
        let trader_id = uuid::Uuid::new_v4();
        let mut portfolio = Portfolio::new(trader_id);
        portfolio.add_cash(1000.0).unwrap();

        let mut market = Market::new(
            "Test".to_string(),
            vec!["A".to_string(), "B".to_string()],
            100.0,
        )
        .unwrap();

        let result = execute_trade(&mut market, trader_id, 0, 0.0, &mut portfolio);
        assert!(matches!(result, Err(TradeError::ZeroAmount)));
    }

    #[test]
    fn test_position_cleanup() {
        let trader_id = uuid::Uuid::new_v4();
        let mut portfolio = Portfolio::new(trader_id);
        portfolio.add_cash(1000.0).unwrap();

        let mut market = Market::new(
            "Test".to_string(),
            vec!["A".to_string(), "B".to_string()],
            100.0,
        )
        .unwrap();

        // Buy then sell all shares
        execute_trade(&mut market, trader_id, 0, 10.0, &mut portfolio).unwrap();

        assert!(portfolio.position(market.id(), 0).is_some());

        execute_trade(&mut market, trader_id, 0, -10.0, &mut portfolio).unwrap();

        // Position should be cleaned up
        assert!(portfolio.position(market.id(), 0).is_none());
        // And the market is flat again.
        assert_eq!(market.quantities()[0], 0.0);
    }
}
