//! Signal processing and market analysis
//!
//! This module provides tools for analyzing market signals, detecting convergence,
//! and processing price time series data.

use crate::MarketId;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// A market signal snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketSignal {
    /// Market identifier
    pub market_id: MarketId,
    /// Current prices for all outcomes
    pub prices: Vec<f64>,
    /// Timestamp of the signal
    pub timestamp: DateTime<Utc>,
    /// Number of trades executed
    pub trade_count: u64,
    /// Total trading volume
    pub volume: f64,
}

impl MarketSignal {
    /// Create a new market signal
    pub fn new(
        market_id: MarketId,
        prices: Vec<f64>,
        trade_count: u64,
        volume: f64,
    ) -> Self {
        Self {
            market_id,
            prices,
            timestamp: Utc::now(),
            trade_count,
            volume,
        }
    }

    /// Calculate signal entropy
    pub fn entropy(&self) -> f64 {
        entropy(&self.prices)
    }

    /// Check if signal represents convergence (low entropy)
    pub fn is_converged(&self, threshold: f64) -> bool {
        self.entropy() < threshold
    }

    /// Get the most likely outcome (highest probability)
    pub fn most_likely_outcome(&self) -> Option<usize> {
        self.prices
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
    }

    /// Get confidence in the most likely outcome
    pub fn confidence(&self) -> f64 {
        self.prices
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .copied()
            .unwrap_or(0.0)
    }
}

/// Time series of market signals
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalSeries {
    signals: VecDeque<MarketSignal>,
    max_length: usize,
}

impl SignalSeries {
    /// Create a new signal series with maximum length
    pub fn new(max_length: usize) -> Self {
        Self {
            signals: VecDeque::new(),
            max_length,
        }
    }

    /// Add a new signal to the series
    pub fn push(&mut self, signal: MarketSignal) {
        self.signals.push_back(signal);
        
        // Maintain maximum length
        while self.signals.len() > self.max_length {
            self.signals.pop_front();
        }
    }

    /// Get the latest signal
    pub fn latest(&self) -> Option<&MarketSignal> {
        self.signals.back()
    }

    /// Get all signals
    pub fn signals(&self) -> &VecDeque<MarketSignal> {
        &self.signals
    }

    /// Calculate convergence over the series
    pub fn convergence(&self) -> f64 {
        convergence(self.signals.iter().collect::<Vec<_>>().as_slice())
    }

    /// Get price history for a specific outcome
    pub fn price_history(&self, outcome: usize) -> Vec<(f64, f64)> {
        price_history(self.signals.iter().collect::<Vec<_>>().as_slice(), outcome)
    }

    /// Calculate price volatility for an outcome
    pub fn volatility(&self, outcome: usize) -> f64 {
        let prices: Vec<f64> = self.signals
            .iter()
            .filter_map(|s| s.prices.get(outcome).copied())
            .collect();
            
        if prices.len() < 2 {
            return 0.0;
        }
        
        let mean = prices.iter().sum::<f64>() / prices.len() as f64;
        let variance = prices.iter()
            .map(|&p| (p - mean).powi(2))
            .sum::<f64>() / prices.len() as f64;
            
        variance.sqrt()
    }

    /// Get trend direction for an outcome (-1 = down, 0 = stable, 1 = up)
    pub fn trend(&self, outcome: usize, window: usize) -> f64 {
        if self.signals.len() < window {
            return 0.0;
        }

        let recent: Vec<f64> = self.signals
            .iter()
            .rev()
            .take(window)
            .filter_map(|s| s.prices.get(outcome).copied())
            .collect();

        if recent.len() < 2 {
            return 0.0;
        }

        let first = recent[recent.len() - 1];
        let last = recent[0];
        
        (last - first).signum()
    }

    /// Check if prices have stabilized (low recent volatility)
    pub fn is_stable(&self, outcome: usize, threshold: f64) -> bool {
        if self.signals.len() < 3 {
            return false;
        }

        let recent_volatility = {
            let recent: Vec<f64> = self.signals
                .iter()
                .rev()
                .take(5)
                .filter_map(|s| s.prices.get(outcome).copied())
                .collect();
            
            if recent.len() < 2 {
                return false;
            }
            
            let mean = recent.iter().sum::<f64>() / recent.len() as f64;
            let variance = recent.iter()
                .map(|&p| (p - mean).powi(2))
                .sum::<f64>() / recent.len() as f64;
                
            variance.sqrt()
        };

        recent_volatility < threshold
    }

    /// Calculate momentum for an outcome
    pub fn momentum(&self, outcome: usize) -> f64 {
        if self.signals.len() < 3 {
            return 0.0;
        }

        let prices: Vec<f64> = self.signals
            .iter()
            .rev()
            .take(3)
            .filter_map(|s| s.prices.get(outcome).copied())
            .collect();

        if prices.len() < 3 {
            return 0.0;
        }

        // Simple momentum: (latest - middle) - (middle - earliest)
        let latest = prices[0];
        let middle = prices[1];
        let earliest = prices[2];
        
        (latest - middle) - (middle - earliest)
    }
}

/// Calculate convergence of market signals
/// 
/// Returns a value between 0.0 and 1.0 where:
/// - 0.0 = highly volatile, not converged
/// - 1.0 = fully converged, stable prices
pub fn convergence(signals: &[&MarketSignal]) -> f64 {
    if signals.len() < 2 {
        return 0.0;
    }

    let num_outcomes = signals[0].prices.len();
    if num_outcomes == 0 {
        return 1.0;
    }

    let mut total_stability = 0.0;

    // Calculate stability for each outcome
    for outcome in 0..num_outcomes {
        let prices: Vec<f64> = signals
            .iter()
            .filter_map(|s| s.prices.get(outcome).copied())
            .collect();

        if prices.len() < 2 {
            continue;
        }

        // Calculate variance
        let mean = prices.iter().sum::<f64>() / prices.len() as f64;
        let variance = prices.iter()
            .map(|&p| (p - mean).powi(2))
            .sum::<f64>() / prices.len() as f64;

        // Convert variance to stability (lower variance = higher stability)
        let stability = (-variance * 10.0).exp();
        total_stability += stability;
    }

    total_stability / num_outcomes as f64
}

/// Extract price history for a specific outcome
/// 
/// Returns a vector of (timestamp, price) pairs
pub fn price_history(signals: &[&MarketSignal], outcome: usize) -> Vec<(f64, f64)> {
    signals
        .iter()
        .filter_map(|signal| {
            signal.prices.get(outcome).map(|&price| {
                let timestamp = signal.timestamp.timestamp() as f64;
                (timestamp, price)
            })
        })
        .collect()
}

/// Calculate Shannon entropy of a probability distribution
/// 
/// Higher values indicate more uncertainty/randomness
pub fn entropy(prices: &[f64]) -> f64 {
    if prices.is_empty() {
        return 0.0;
    }

    // Normalize to ensure valid probabilities
    let sum: f64 = prices.iter().sum();
    if sum <= 0.0 {
        return 0.0;
    }

    let mut entropy = 0.0;
    for &price in prices {
        let p = price / sum;
        if p > 0.0 {
            entropy -= p * p.ln();
        }
    }

    entropy
}

/// Calculate maximum possible entropy for given number of outcomes
pub fn max_entropy(num_outcomes: usize) -> f64 {
    if num_outcomes <= 1 {
        return 0.0;
    }
    
    (num_outcomes as f64).ln()
}

/// Calculate normalized entropy (0.0 to 1.0)
pub fn normalized_entropy(prices: &[f64]) -> f64 {
    let ent = entropy(prices);
    let max_ent = max_entropy(prices.len());
    
    if max_ent == 0.0 {
        0.0
    } else {
        ent / max_ent
    }
}

/// Calculate market efficiency metric
/// 
/// Compares trading volume to price movement
pub fn efficiency(signals: &[&MarketSignal]) -> f64 {
    if signals.len() < 2 {
        return 0.0;
    }

    let total_volume: f64 = signals.iter().map(|s| s.volume).sum();
    if total_volume == 0.0 {
        return 0.0;
    }

    // Calculate total price movement
    let mut total_movement = 0.0;
    for i in 1..signals.len() {
        for j in 0..signals[i].prices.len() {
            if let (Some(prev), Some(curr)) = 
                (signals[i-1].prices.get(j), signals[i].prices.get(j)) {
                total_movement += (curr - prev).abs();
            }
        }
    }

    // Efficiency is movement per unit volume
    if total_volume > 0.0 {
        total_movement / total_volume
    } else {
        0.0
    }
}

/// Detect market anomalies (sudden large price movements)
pub fn detect_anomalies(signals: &[&MarketSignal], threshold: f64) -> Vec<usize> {
    if signals.len() < 2 {
        return Vec::new();
    }

    let mut anomalies = Vec::new();

    for i in 1..signals.len() {
        let prev = &signals[i-1];
        let curr = &signals[i];

        for j in 0..curr.prices.len() {
            if let (Some(&prev_price), Some(&curr_price)) = 
                (prev.prices.get(j), curr.prices.get(j)) {
                let change = (curr_price - prev_price).abs();
                if change > threshold {
                    anomalies.push(i);
                    break;
                }
            }
        }
    }

    anomalies
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::{assert_relative_eq, assert_abs_diff_eq};
    use uuid::Uuid;

    #[test]
    fn test_market_signal_creation() {
        let market_id = Uuid::new_v4();
        let prices = vec![0.3, 0.7];
        let signal = MarketSignal::new(market_id, prices.clone(), 10, 100.0);

        assert_eq!(signal.market_id, market_id);
        assert_eq!(signal.prices, prices);
        assert_eq!(signal.trade_count, 10);
        assert_eq!(signal.volume, 100.0);
    }

    #[test]
    fn test_signal_entropy() {
        let market_id = Uuid::new_v4();
        
        // Equal probabilities should have high entropy
        let equal_signal = MarketSignal::new(market_id, vec![0.5, 0.5], 0, 0.0);
        let equal_entropy = equal_signal.entropy();
        assert_relative_eq!(equal_entropy, 2.0_f64.ln(), epsilon = 1e-10);

        // Skewed probabilities should have lower entropy
        let skewed_signal = MarketSignal::new(market_id, vec![0.9, 0.1], 0, 0.0);
        let skewed_entropy = skewed_signal.entropy();
        assert!(skewed_entropy < equal_entropy);
    }

    #[test]
    fn test_signal_most_likely_outcome() {
        let market_id = Uuid::new_v4();
        let signal = MarketSignal::new(market_id, vec![0.2, 0.7, 0.1], 0, 0.0);

        assert_eq!(signal.most_likely_outcome(), Some(1));
        assert_relative_eq!(signal.confidence(), 0.7, epsilon = 1e-10);
    }

    #[test]
    fn test_signal_convergence() {
        let market_id = Uuid::new_v4();
        
        let converged = MarketSignal::new(market_id, vec![0.95, 0.05], 0, 0.0);
        assert!(converged.is_converged(0.5));

        let unconverged = MarketSignal::new(market_id, vec![0.6, 0.4], 0, 0.0);
        assert!(!unconverged.is_converged(0.5));
    }

    #[test]
    fn test_signal_series() {
        let mut series = SignalSeries::new(3);
        let market_id = Uuid::new_v4();

        // Add signals
        for i in 0..5 {
            let signal = MarketSignal::new(market_id, vec![0.5, 0.5], i, i as f64);
            series.push(signal);
        }

        // Should only keep last 3
        assert_eq!(series.signals().len(), 3);
        
        let latest = series.latest().unwrap();
        assert_eq!(latest.trade_count, 4);
    }

    #[test]
    fn test_entropy_calculation() {
        // Equal probabilities
        let equal = vec![0.5, 0.5];
        let equal_entropy = entropy(&equal);
        assert_relative_eq!(equal_entropy, 2.0_f64.ln(), epsilon = 1e-10);

        // Certain outcome
        let certain = vec![1.0, 0.0];
        let certain_entropy = entropy(&certain);
        assert_abs_diff_eq!(certain_entropy, 0.0, epsilon = 1e-10);

        // Three equal outcomes
        let three_equal = vec![1.0/3.0, 1.0/3.0, 1.0/3.0];
        let three_entropy = entropy(&three_equal);
        assert_relative_eq!(three_entropy, 3.0_f64.ln(), epsilon = 1e-10);
    }

    #[test]
    fn test_max_entropy() {
        assert_eq!(max_entropy(1), 0.0);
        assert_relative_eq!(max_entropy(2), 2.0_f64.ln(), epsilon = 1e-10);
        assert_relative_eq!(max_entropy(3), 3.0_f64.ln(), epsilon = 1e-10);
    }

    #[test]
    fn test_normalized_entropy() {
        let equal = vec![0.5, 0.5];
        assert_relative_eq!(normalized_entropy(&equal), 1.0, epsilon = 1e-10);

        let certain = vec![1.0, 0.0];
        assert_abs_diff_eq!(normalized_entropy(&certain), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_convergence_calculation() {
        let market_id = Uuid::new_v4();
        
        // Stable signals should show high convergence
        let signals: Vec<MarketSignal> = (0..5)
            .map(|i| MarketSignal::new(market_id, vec![0.6, 0.4], i, i as f64))
            .collect();
        let signal_refs: Vec<&MarketSignal> = signals.iter().collect();
        
        let stable_convergence = convergence(&signal_refs);
        
        // Volatile signals should show low convergence
        let volatile_signals: Vec<MarketSignal> = (0..5)
            .map(|i| {
                let price = 0.5 + 0.3 * (i as f64 * 0.5).sin();
                MarketSignal::new(market_id, vec![price, 1.0 - price], i, i as f64)
            })
            .collect();
        let volatile_refs: Vec<&MarketSignal> = volatile_signals.iter().collect();
        
        let volatile_convergence = convergence(&volatile_refs);
        
        assert!(stable_convergence > volatile_convergence);
    }

    #[test]
    fn test_price_history() {
        let market_id = Uuid::new_v4();
        let signals: Vec<MarketSignal> = (0..3)
            .map(|i| {
                let mut signal = MarketSignal::new(market_id, vec![i as f64 * 0.1, 1.0 - i as f64 * 0.1], i, i as f64);
                signal.timestamp = DateTime::from_timestamp(1000 + i as i64, 0).unwrap();
                signal
            })
            .collect();
        let signal_refs: Vec<&MarketSignal> = signals.iter().collect();
        
        let history = price_history(&signal_refs, 0);
        assert_eq!(history.len(), 3);
        
        // Check timestamps are increasing
        assert!(history[1].0 > history[0].0);
        assert!(history[2].0 > history[1].0);
        
        // Check prices are increasing
        assert!(history[1].1 > history[0].1);
        assert!(history[2].1 > history[1].1);
    }

    #[test]
    fn test_signal_series_volatility() {
        let mut series = SignalSeries::new(10);
        let market_id = Uuid::new_v4();

        // Add signals with varying prices
        let prices = vec![0.5, 0.6, 0.4, 0.7, 0.3];
        for (i, &price) in prices.iter().enumerate() {
            let signal = MarketSignal::new(market_id, vec![price, 1.0 - price], i as u64, i as f64);
            series.push(signal);
        }

        let volatility = series.volatility(0);
        assert!(volatility > 0.0); // Should have some volatility

        // Add more stable signals
        for i in 5..10 {
            let signal = MarketSignal::new(market_id, vec![0.5, 0.5], i as u64, i as f64);
            series.push(signal);
        }

        let new_volatility = series.volatility(0);
        assert!(new_volatility < volatility); // Should be less volatile now
    }

    #[test]
    fn test_signal_series_trend() {
        let mut series = SignalSeries::new(10);
        let market_id = Uuid::new_v4();

        // Add upward trending signals
        for i in 0..5 {
            let price = 0.3 + i as f64 * 0.1;
            let signal = MarketSignal::new(market_id, vec![price, 1.0 - price], i as u64, i as f64);
            series.push(signal);
        }

        let trend = series.trend(0, 3);
        assert!(trend > 0.0); // Should detect upward trend

        // Add downward trending signals
        for i in 0..3 {
            let price = 0.7 - i as f64 * 0.1;
            let signal = MarketSignal::new(market_id, vec![price, 1.0 - price], (i + 5) as u64, (i + 5) as f64);
            series.push(signal);
        }

        let new_trend = series.trend(0, 3);
        assert!(new_trend < 0.0); // Should detect downward trend
    }

    #[test]
    fn test_signal_series_stability() {
        let mut series = SignalSeries::new(10);
        let market_id = Uuid::new_v4();

        // Add stable signals
        for i in 0..5 {
            let signal = MarketSignal::new(market_id, vec![0.6, 0.4], i as u64, i as f64);
            series.push(signal);
        }

        assert!(series.is_stable(0, 0.1)); // Should be stable

        // Add a volatile signal
        let volatile_signal = MarketSignal::new(market_id, vec![0.9, 0.1], 5, 5.0);
        series.push(volatile_signal);

        assert!(!series.is_stable(0, 0.1)); // Should not be stable anymore
    }

    #[test]
    fn test_detect_anomalies() {
        let market_id = Uuid::new_v4();
        let mut signals = Vec::new();

        // Normal signals
        for i in 0..3 {
            signals.push(MarketSignal::new(market_id, vec![0.5, 0.5], i, i as f64));
        }

        // Anomalous signal (large jump from 0.5 to 0.9)
        signals.push(MarketSignal::new(market_id, vec![0.9, 0.1], 3, 3.0));

        // Return to normal 
        signals.push(MarketSignal::new(market_id, vec![0.5, 0.5], 4, 4.0));

        let signal_refs: Vec<&MarketSignal> = signals.iter().collect();
        let anomalies = detect_anomalies(&signal_refs, 0.35); // Higher threshold to catch only big jumps

        assert_eq!(anomalies.len(), 2); // Spike up at index 3, spike down at index 4
        assert_eq!(anomalies[0], 3); // Index of first anomalous signal
        assert_eq!(anomalies[1], 4); // Index of second anomalous signal
    }

    #[test]
    fn test_efficiency_calculation() {
        let market_id = Uuid::new_v4();
        let signals: Vec<MarketSignal> = vec![
            MarketSignal::new(market_id, vec![0.5, 0.5], 0, 10.0),
            MarketSignal::new(market_id, vec![0.6, 0.4], 1, 20.0),
            MarketSignal::new(market_id, vec![0.7, 0.3], 2, 30.0),
        ];
        let signal_refs: Vec<&MarketSignal> = signals.iter().collect();
        
        let eff = efficiency(&signal_refs);
        assert!(eff > 0.0); // Should have some efficiency measure
    }

    #[test]
    fn test_signal_series_momentum() {
        let mut series = SignalSeries::new(10);
        let market_id = Uuid::new_v4();

        // Add signals with accelerating price change
        let prices = vec![0.3, 0.4, 0.6]; // Accelerating upward
        for (i, &price) in prices.iter().enumerate() {
            let signal = MarketSignal::new(market_id, vec![price, 1.0 - price], i as u64, i as f64);
            series.push(signal);
        }

        let momentum = series.momentum(0);
        assert!(momentum > 0.0); // Positive momentum

        // Add signals with decelerating price change
        series.push(MarketSignal::new(market_id, vec![0.65, 0.35], 3, 3.0)); // Smaller increase

        let new_momentum = series.momentum(0);
        assert!(new_momentum < momentum); // Should have lower momentum
    }
}