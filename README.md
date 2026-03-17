# ghostsignals

Prediction markets as collective intelligence using the Logarithmic Market Scoring Rule (LMSR).

[![Crates.io](https://img.shields.io/crates/v/ghostsignals.svg)](https://crates.io/crates/ghostsignals)
[![Documentation](https://docs.rs/ghostsignals/badge.svg)](https://docs.rs/ghostsignals)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> *"When agents trade on what they believe, the market price converges to the collective's true estimate — emergence from interference."*

Part of the [ghostmagicOS](https://github.com/ghostmagicOS) ecosystem: `dx/dt = f(x) - Iηx`  
The market IS the interference term.

## Overview

ghostsignals implements Robin Hanson's Logarithmic Market Scoring Rule (LMSR) for creating prediction markets that aggregate collective intelligence. The library provides:

- **Pure Rust implementation** with no async dependencies
- **Numerically stable** LMSR calculations using the log-sum-exp trick
- **Complete market lifecycle** management (creation, trading, resolution)
- **Portfolio tracking** and position management
- **Signal processing** for convergence detection and price analysis
- **Comprehensive error handling** with no panics in library code

## Quick Start

```rust
use ghostsignals::{Engine, MarketError};
use uuid::Uuid;

fn main() -> Result<(), MarketError> {
    // Create a prediction market engine
    let mut engine = Engine::new();
    
    // Create a market
    let market_id = engine.create_market(
        "Will it rain tomorrow?".to_string(),
        vec!["Yes".to_string(), "No".to_string()],
        100.0, // liquidity parameter
    )?;
    
    // Check initial prices (should be ~50/50)
    let prices = engine.prices(market_id)?;
    println!("Initial prices: Yes={:.2}, No={:.2}", prices[0], prices[1]);
    
    // A trader makes a prediction
    let trader = Uuid::new_v4();
    let trade = engine.trade(market_id, trader, 0, 10.0)?; // Buy 10 shares of "Yes"
    
    // Prices update based on the trade
    let new_prices = engine.prices(market_id)?;
    println!("After trade: Yes={:.2}, No={:.2}", new_prices[0], new_prices[1]);
    
    // Get market signal for analysis
    let signal = engine.signal(market_id)?;
    println!("Market entropy: {:.3}", signal.entropy());
    
    Ok(())
}
```

## Core Concepts

### LMSR Mathematics

The Logarithmic Market Scoring Rule uses these key formulas:

- **Cost function**: `C(q) = b * ln(Σ exp(qᵢ/b))`
- **Prices**: `p(i) = exp(qᵢ/b) / Σ exp(qⱼ/b)`
- **Trade cost**: `C(q_after) - C(q_before)`

Where:
- `q` is the vector of quantities for each outcome
- `b` is the liquidity parameter (higher = less price sensitivity)
- Prices always sum to 1.0 (probability constraint)

### Market Lifecycle

```rust
use ghostsignals::{Engine, MarketState};

let mut engine = Engine::new();

// 1. Create market
let market_id = engine.create_market(
    "Who will win the election?".to_string(),
    vec!["Alice".to_string(), "Bob".to_string(), "Charlie".to_string()],
    200.0,
)?;

// 2. Trading phase
let alice_supporter = Uuid::new_v4();
engine.trade(market_id, alice_supporter, 0, 50.0)?; // Buy Alice shares

let bob_supporter = Uuid::new_v4();
engine.trade(market_id, bob_supporter, 1, 30.0)?; // Buy Bob shares

// 3. Resolution
engine.resolve(market_id, 0)?; // Alice wins!
```

### Portfolio Management

```rust
use ghostsignals::Engine;

let mut engine = Engine::new();
let market_id = engine.create_market(/* ... */)?;

let trader = Uuid::new_v4();

// Make some trades
engine.trade(market_id, trader, 0, 20.0)?;
engine.trade(market_id, trader, 1, 10.0)?;

// Check portfolio
if let Some(portfolio) = engine.portfolio(trader) {
    println!("Trader has {} cash", portfolio.cash);
    
    if let Some(position) = portfolio.position(market_id, 0) {
        println!("Position: {} shares at avg cost {:.3}", 
                 position.shares, position.avg_cost);
    }
    
    // Calculate current value
    if let Some(market) = engine.market(market_id) {
        let value = portfolio.market_value(market);
        println!("Portfolio value: {:.2}", value);
    }
}
```

### Signal Processing

```rust
use ghostsignals::{signals::SignalSeries, Engine};

let mut engine = Engine::new();
let market_id = engine.create_market(/* ... */)?;

let mut series = SignalSeries::new(100); // Keep last 100 signals

// Collect signals over time
for _ in 0..50 {
    // ... trades happen ...
    
    let signal = engine.signal(market_id)?;
    series.push(signal);
}

// Analyze convergence
let convergence = series.convergence();
println!("Market convergence: {:.3}", convergence);

// Check price volatility
let volatility = series.volatility(0); // For outcome 0
println!("Price volatility: {:.3}", volatility);

// Get price history
let history = series.price_history(0);
for (timestamp, price) in history.iter().take(5) {
    println!("Time {}: Price {:.3}", timestamp, price);
}
```

## Advanced Usage

### Custom Error Handling

```rust
use ghostsignals::{Engine, TradeError, MarketError};

let mut engine = Engine::new();
let market_id = engine.create_market(/* ... */)?;

let trader = Uuid::new_v4();

match engine.trade(market_id, trader, 0, 1000000.0) {
    Ok(trade) => println!("Trade executed: cost {:.2}", trade.cost),
    Err(TradeError::InsufficientFunds { required, available }) => {
        println!("Not enough cash: need {:.2}, have {:.2}", required, available);
    },
    Err(e) => println!("Trade failed: {}", e),
}
```

### Market Analysis

```rust
use ghostsignals::{signals, Engine};

let mut engine = Engine::new();
let market_id = engine.create_market(/* ... */)?;

// Get market entropy (uncertainty measure)
let signal = engine.signal(market_id)?;
let entropy = signal.entropy();
let max_entropy = signals::max_entropy(signal.prices.len());
let normalized_entropy = entropy / max_entropy;

println!("Market uncertainty: {:.1}%", normalized_entropy * 100.0);

// Check if market has converged
if signal.is_converged(0.3) {
    println!("Market has converged on outcome: {:?}", 
             signal.most_likely_outcome());
}
```

### Working with Different Market Types

```rust
use ghostsignals::Engine;

// Binary market
let binary_id = engine.create_market(
    "Will GDP grow this quarter?".to_string(),
    vec!["Yes".to_string(), "No".to_string()],
    100.0,
)?;

// Multi-outcome market
let multi_id = engine.create_market(
    "Which team will win the World Cup?".to_string(),
    vec![
        "Brazil".to_string(),
        "Germany".to_string(), 
        "Spain".to_string(),
        "France".to_string(),
        "Other".to_string(),
    ],
    500.0, // Higher liquidity for more outcomes
)?;

// Categorical market
let category_id = engine.create_market(
    "What will be the weather tomorrow?".to_string(),
    vec![
        "Sunny".to_string(),
        "Cloudy".to_string(),
        "Rainy".to_string(),
        "Snowy".to_string(),
    ],
    200.0,
)?;
```

## Technical Details

### Numerical Stability

ghostsignals uses the log-sum-exp trick throughout to prevent overflow:

```rust
// Instead of: ln(Σ exp(xᵢ))
// We compute: max(xᵢ) + ln(Σ exp(xᵢ - max(xᵢ)))
```

This ensures stable calculations even with large quantity values.

### Memory Safety

- No `unwrap()` calls in library code
- All public functions return `Result` types
- Comprehensive error types with `thiserror`
- No unsafe code

### Performance

- Pure computation, no I/O or async
- Efficient data structures
- Minimal allocations in hot paths
- O(n) complexity for most operations where n = number of outcomes

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
ghostsignals = "0.1.0"
```

## Features

- `serde` - Serialization support (enabled by default)
- No optional features currently - library is lightweight by design

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Related Work

- [Robin Hanson's LMSR paper](https://mason.gmu.edu/~rhanson/mktscore.pdf)
- [Prediction Markets: An Extended Bibliography](https://mason.gmu.edu/~rhanson/bibtex.html)
- [ghostmagicOS ecosystem](https://github.com/ghostmagicOS)

## Citation

If you use ghostsignals in academic work, please cite:

```bibtex
@software{ghostsignals,
  title = {ghostsignals: Prediction Markets as Collective Intelligence},
  author = {ghostmagicOS},
  year = {2026},
  url = {https://github.com/NickFlach/ghostsignals-rs}
}
```

---

*Part of the ghostmagicOS project: turning collective intelligence into executable reality.*