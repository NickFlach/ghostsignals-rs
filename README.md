# ghostsignals

Prediction markets as collective intelligence using the Logarithmic Market Scoring Rule (LMSR).

[![Crates.io](https://img.shields.io/crates/v/ghostsignals.svg)](https://crates.io/crates/ghostsignals)
[![Documentation](https://docs.rs/ghostsignals/badge.svg)](https://docs.rs/ghostsignals)
[![License](https://img.shields.io/badge/license-Space%20Child%20v1.0-blueviolet)]()

> *"When agents trade on what they believe, the market price converges to the collective's true estimate — emergence from interference."*

Part of the [ghostmagicOS](https://github.com/ghostmagicOS) ecosystem: `dx/dt = f(x) - Iηx`  
The market IS the interference term.

## Overview

ghostsignals implements Robin Hanson's Logarithmic Market Scoring Rule (LMSR) for creating prediction markets that aggregate collective intelligence. The library provides:

- **Pure Rust implementation** with no async dependencies
- **Numerically stable** LMSR calculations using the log-sum-exp trick and a cancellation-free trade cost
- **Complete market lifecycle** management (creation, trading, resolution, settlement)
- **Portfolio tracking** and position management
- **Signal processing** for convergence detection and price analysis
- **Comprehensive error handling** with no panics in library code
- **Enforced invariants**: validated inputs, conserved money, bounded market-maker loss, validated deserialization

Every example in this README is compiled and run as a doctest.

## Quick Start

```rust
use ghostsignals::{Engine, TradeError};
use uuid::Uuid;

fn main() -> Result<(), TradeError> {
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

    // A trader needs cash before they can buy
    let trader = Uuid::new_v4();
    engine.deposit(trader, 100.0)?;

    // The trader makes a prediction
    let trade = engine.trade(market_id, trader, 0, 10.0)?; // Buy 10 shares of "Yes"
    println!("Paid {:.3} for 10 shares", trade.cost);

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

The low-level functions are available directly:

```rust
use ghostsignals::lmsr::{cost, prices, trade_cost};

let q = [10.0, 0.0];
let b = 100.0;
let c = cost(&q, b).unwrap();
let p = prices(&q, b).unwrap();
let buy_five = trade_cost(&q, b, 0, 5.0).unwrap();

assert!((p[0] + p[1] - 1.0).abs() < 1e-12);
assert!(p[0] > p[1]);
assert!(buy_five > 0.0 && buy_five <= 5.0);
assert!(c >= 10.0);
```

### Market Lifecycle

```rust
use ghostsignals::{Engine, MarketState};
use uuid::Uuid;

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let mut engine = Engine::new();

// 1. Create market
let market_id = engine.create_market(
    "Who will win the election?".to_string(),
    vec!["Alice".to_string(), "Bob".to_string(), "Charlie".to_string()],
    200.0,
)?;

// 2. Trading phase
let alice_supporter = Uuid::new_v4();
engine.deposit(alice_supporter, 500.0)?;
engine.trade(market_id, alice_supporter, 0, 50.0)?; // Buy Alice shares

let bob_supporter = Uuid::new_v4();
engine.deposit(bob_supporter, 500.0)?;
engine.trade(market_id, bob_supporter, 1, 30.0)?; // Buy Bob shares

// 3. Resolution
engine.resolve(market_id, 0)?; // Alice wins!
assert_eq!(engine.market(market_id).unwrap().state(), MarketState::Resolved);

// 4. Settlement: one unit per winning share, positions cleared
let payouts = engine.settle(market_id)?;
assert_eq!(payouts, vec![(alice_supporter, 50.0)]);

// Resolving again to the same outcome is a no-op; settling again pays nothing
engine.resolve(market_id, 0)?;
assert!(engine.settle(market_id)?.is_empty());
# Ok(())
# }
```

### Portfolio Management

```rust
use ghostsignals::Engine;
use uuid::Uuid;

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let mut engine = Engine::new();
let market_id = engine.create_market(
    "Test".to_string(),
    vec!["A".to_string(), "B".to_string()],
    100.0,
)?;

let trader = Uuid::new_v4();
engine.deposit(trader, 100.0)?;

// Make some trades
engine.trade(market_id, trader, 0, 20.0)?;
engine.trade(market_id, trader, 1, 10.0)?;

// Check portfolio
if let Some(portfolio) = engine.portfolio(trader) {
    println!("Trader has {} cash", portfolio.cash());

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
# Ok(())
# }
```

### Signal Processing

```rust
use ghostsignals::{signals::SignalSeries, Engine};
use uuid::Uuid;

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let mut engine = Engine::new();
let market_id = engine.create_market(
    "Test".to_string(),
    vec!["A".to_string(), "B".to_string()],
    100.0,
)?;
let trader = Uuid::new_v4();
engine.deposit(trader, 1_000.0)?;

let mut series = SignalSeries::new(100); // Keep last 100 signals

// Collect signals over time
for i in 0..50 {
    engine.trade(market_id, trader, i % 2, 1.0)?;

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
# Ok(())
# }
```

## Advanced Usage

### Custom Error Handling

```rust
use ghostsignals::{Engine, TradeError, MarketError};
use uuid::Uuid;

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let mut engine = Engine::new();
let market_id = engine.create_market(
    "Test".to_string(),
    vec!["A".to_string(), "B".to_string()],
    100.0,
)?;

let trader = Uuid::new_v4();
engine.deposit(trader, 10.0)?;

match engine.trade(market_id, trader, 0, 1000000.0) {
    Ok(trade) => println!("Trade executed: cost {:.2}", trade.cost),
    Err(TradeError::InsufficientFunds { required, available }) => {
        println!("Not enough cash: need {:.2}, have {:.2}", required, available);
    },
    Err(e) => println!("Trade failed: {}", e),
}

// Invalid inputs are typed errors, never panics or NaN
assert!(matches!(
    engine.trade(market_id, trader, 0, f64::NAN),
    Err(TradeError::InvalidAmount(_))
));
assert!(matches!(
    engine.create_market("Q?".to_string(), vec!["A".to_string(), "a".to_string()], 10.0),
    Err(MarketError::DuplicateOutcomeLabel(_))
));
# Ok(())
# }
```

### Market Analysis

```rust
use ghostsignals::{signals, Engine};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let mut engine = Engine::new();
let market_id = engine.create_market(
    "Test".to_string(),
    vec!["A".to_string(), "B".to_string()],
    100.0,
)?;

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
# Ok(())
# }
```

### Working with Different Market Types

```rust
use ghostsignals::Engine;

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let mut engine = Engine::new();

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

assert_eq!(engine.prices(multi_id)?.len(), 5);
assert_ne!(binary_id, category_id);
# Ok(())
# }
```

## Technical Details

### Numerical Stability

ghostsignals uses the log-sum-exp trick throughout to prevent overflow:

```text
// Instead of: ln(Σ exp(xᵢ))
// We compute: max(xᵢ) + ln(Σ exp(xᵢ - max(xᵢ)))
```

Trade costs are not computed as the difference of two cost-function
evaluations (which cancels catastrophically for small trades). Instead the
closed form `b · ln(1 + pᵢ · (exp(Δ/b) − 1))` is evaluated with `ln_1p` and
`exp_m1`, falling back to the direct log-sum-exp difference only where that
form loses accuracy (large sells of a dominant outcome, buys past `exp`
overflow). A trade of `1e-9` shares is priced to better than one part in a
million; a trade of `5000 · b` shares is priced to one part in `1e12`.

### Guaranteed Invariants

- The liquidity `b` is finite and positive; quantities and amounts are finite.
  Anything else is a typed error, never NaN or infinity.
- Prices are finite, in `[0, 1]`, sum to 1 within rounding, and are monotone in
  the traded quantity.
- A `Market` always satisfies `Market::validate()`, including after
  deserialization; an invalid serialized market is refused.
- Outstanding shares per outcome never go negative; a trader can never sell
  more than they hold, and can always close a whole position even after
  floating-point dust has accumulated.
- Cash balances are finite and never negative.
- Money is conserved: `Σ balances + Σ Market::pool()` is constant across any
  trade sequence, and at settlement the market maker's loss is at most
  `b · ln(n)`.
- Resolution is idempotent for the same outcome and refused for a different
  one; trades after resolution are refused; settlement pays once.

These are checked by property-based tests (`proptest`) in `tests/`.

### Memory Safety

- No `unwrap()` on user-controlled data in library code
- All fallible public functions return `Result` types
- Comprehensive error types with `thiserror` (marked `#[non_exhaustive]`)
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
ghostsignals = "0.2.0"
```

## Changes in 0.2.0

Breaking:

- `Portfolio::cash` is a private field; read it with `Portfolio::cash()`.
- `Portfolio::add_cash` returns `Result` and refuses negative or non-finite amounts.
- `Position::apply_trade` returns `Result`; an over-sell or a cost with the wrong sign is an error instead of a silent no-op.
- `Market::resolve` on an already-resolved market succeeds for the same outcome (was an error) and still errors for a different one.
- Error enums are `#[non_exhaustive]` and gained variants: `LmsrError::NonFinite`, `MarketError::{EmptyOutcomeLabel, DuplicateOutcomeLabel, InvalidMarketState, MarketNotResolved}`, `TradeError::InvalidAmount`.
- `Market::new` rejects NaN/infinite liquidity, blank outcome labels and duplicate labels (case-insensitive).
- `Market` and `Portfolio` deserialization validates the value and can fail.
- `execute_trade`/`Engine::trade` on a closed market return `TradeError::MarketClosed` (was `TradeError::MarketError(MarketError::MarketClosed)`), refuse NaN/infinite amounts, and may round a near-whole-position sell to the exact position.
- `signals::efficiency` divides by the volume traded across the window (`last - first`) instead of the sum of cumulative snapshots.
- `SignalSeries::new(0)` keeps one signal instead of none.

Added:

- `Engine::deposit`, `Engine::settle`, `Engine::markets`
- `Market::pool`, `Market::validate`
- `pub mod lmsr / market / trading / signals` alongside the flat re-exports
- Serialized markets gain a `pool` field (older files load with `pool = 0`)

## Features

- `serde` - Serialization support (enabled by default)
- No optional features currently - library is lightweight by design

## License

This project is licensed under the Space Child License v1.0 — see [LICENSE](./LICENSE) for the terms.

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
