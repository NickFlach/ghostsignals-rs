//! Edge-case tests for invariants the public API promises.
//!
//! Each test names the invariant it guards. Float comparisons use explicit
//! epsilons with a comment saying where the epsilon comes from.

use approx::assert_relative_eq;
use ghostsignals::{
    convergence, cost, efficiency, entropy, execute_trade, prices, trade_cost, Engine,
    LmsrError, Market, MarketError, MarketSignal, MarketState, Portfolio, SignalSeries,
    TradeError,
};
use uuid::Uuid;

fn binary_market(b: f64) -> Market {
    Market::new("Test?".to_string(), vec!["Yes".into(), "No".into()], b).unwrap()
}

// ---------------------------------------------------------------------------
// LMSR core: b > 0 and finite; NaN / infinity never escape a public fn
// ---------------------------------------------------------------------------

#[test]
fn lmsr_rejects_non_finite_liquidity() {
    for b in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(cost(&[0.0, 0.0], b).is_err(), "cost accepted b={b}");
        assert!(prices(&[0.0, 0.0], b).is_err(), "prices accepted b={b}");
        assert!(
            trade_cost(&[0.0, 0.0], b, 0, 1.0).is_err(),
            "trade_cost accepted b={b}"
        );
    }
}

#[test]
fn lmsr_never_returns_non_finite() {
    let bad_q: [&[f64]; 4] = [
        &[f64::NAN, 0.0],
        &[f64::INFINITY, 0.0],
        &[f64::NEG_INFINITY, f64::NEG_INFINITY],
        &[f64::NEG_INFINITY, 0.0],
    ];
    for q in bad_q {
        if let Ok(c) = cost(q, 1.0) {
            assert!(c.is_finite(), "cost({q:?}) returned non-finite {c}");
        }
        if let Ok(p) = prices(q, 1.0) {
            assert!(p.iter().all(|x| x.is_finite()), "prices({q:?}) = {p:?}");
        }
    }
    // Non-finite trade amounts must be refused, not folded into the state.
    for amount in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(trade_cost(&[0.0, 0.0], 1.0, 0, amount).is_err());
    }
}

#[test]
fn lmsr_rejects_all_negative_infinity_quantities() {
    // The old implementation returned Ok(-inf) here.
    assert!(matches!(
        cost(&[f64::NEG_INFINITY, f64::NEG_INFINITY], 1.0),
        Err(LmsrError::NonFinite { .. })
    ));
}

#[test]
fn small_trade_cost_matches_marginal_price() {
    // For a tiny trade Δ, C(q+Δe_i) - C(q) = p_i·Δ + O(Δ²/b).
    // Direct subtraction of two ~69.3 values loses ~5 digits at Δ=1e-9;
    // the relative tolerance of 1e-6 is far looser than the second-order
    // term (Δ/b = 1e-11) so only cancellation error can fail this.
    let q = [0.0, 0.0];
    let b = 100.0;
    let delta = 1e-9;
    let p = prices(&q, b).unwrap()[0];
    let c = trade_cost(&q, b, 0, delta).unwrap();
    assert_relative_eq!(c, p * delta, max_relative = 1e-6);
}

#[test]
fn large_trades_are_still_computed_by_stable_path() {
    // Buying 5000 shares at b = 1 pushes q/b far past exp() overflow.
    let q = [0.0, 0.0];
    let c = trade_cost(&q, 1.0, 0, 5000.0).unwrap();
    // C(q+Δ) ≈ Δ when p_i -> 1; C(0) = ln 2.  Relative 1e-12: pure f64
    // rounding of numbers of magnitude 5e3 is ~1e-13.
    assert_relative_eq!(c, 5000.0 - 2.0_f64.ln(), max_relative = 1e-12);
    // Selling it all back nets the same magnitude.
    let back = trade_cost(&[5000.0, 0.0], 1.0, 0, -5000.0).unwrap();
    assert_relative_eq!(back, -(5000.0 - 2.0_f64.ln()), max_relative = 1e-12);
}

// ---------------------------------------------------------------------------
// Market construction
// ---------------------------------------------------------------------------

#[test]
fn market_rejects_non_finite_liquidity() {
    for b in [f64::NAN, f64::INFINITY] {
        assert!(
            matches!(
                Market::new("Q?".into(), vec!["A".into(), "B".into()], b),
                Err(MarketError::InvalidLiquidity(_))
            ),
            "Market::new accepted b={b}"
        );
    }
}

#[test]
fn market_rejects_empty_and_duplicate_outcome_labels() {
    assert!(Market::new("Q?".into(), vec!["A".into(), "  ".into()], 10.0).is_err());
    // find_outcome is case-insensitive, so labels must be unique ignoring case.
    assert!(Market::new("Q?".into(), vec!["Yes".into(), "yes".into()], 10.0).is_err());
    assert!(Market::new("Q?".into(), vec!["A".into(), "B".into(), "A".into()], 10.0).is_err());
}

// ---------------------------------------------------------------------------
// Signals: no panics on hostile input, no NaN out
// ---------------------------------------------------------------------------

#[test]
fn signal_with_nan_prices_does_not_panic() {
    let s = MarketSignal::new(Uuid::new_v4(), vec![f64::NAN, 0.5, 0.2], 0, 0.0);
    // Must not panic; the NaN entry is simply never "most likely".
    assert_eq!(s.most_likely_outcome(), Some(1));
    assert_relative_eq!(s.confidence(), 0.5, epsilon = 0.0);
    let all_nan = MarketSignal::new(Uuid::new_v4(), vec![f64::NAN, f64::NAN], 0, 0.0);
    assert_eq!(all_nan.most_likely_outcome(), None);
    assert_eq!(all_nan.confidence(), 0.0);
}

#[test]
fn entropy_is_scale_invariant_at_extremes() {
    // Two equal weights must give ln 2 regardless of magnitude; summing
    // 1e308 + 1e308 overflows unless the inputs are rescaled first.
    assert_relative_eq!(entropy(&[1e308, 1e308]), 2.0_f64.ln(), epsilon = 1e-12);
    assert_relative_eq!(entropy(&[1e-308, 1e-308]), 2.0_f64.ln(), epsilon = 1e-12);
    // Non-finite / negative entries never produce NaN.
    for p in [[f64::NAN, 0.5], [f64::INFINITY, 1.0], [-1.0, 0.5]] {
        let e = entropy(&p);
        assert!(e.is_finite() && e >= 0.0, "entropy({p:?}) = {e}");
    }
}

#[test]
fn convergence_never_returns_nan() {
    let id = Uuid::new_v4();
    let a = MarketSignal::new(id, vec![0.5, 0.5], 0, 0.0);
    let b = MarketSignal::new(id, vec![f64::NAN, 0.5], 1, 0.0);
    let c = convergence(&[&a, &b]);
    assert!(c.is_finite() && (0.0..=1.0).contains(&c), "convergence = {c}");
}

#[test]
fn efficiency_uses_volume_delta_not_cumulative_sum() {
    // MarketSignal.volume is the market's cumulative volume, so the volume
    // traded across the window is last - first, not the sum of snapshots.
    let id = Uuid::new_v4();
    let s = [
        MarketSignal::new(id, vec![0.5, 0.5], 0, 10.0),
        MarketSignal::new(id, vec![0.6, 0.4], 1, 20.0),
        MarketSignal::new(id, vec![0.7, 0.3], 2, 30.0),
    ];
    let refs: Vec<&MarketSignal> = s.iter().collect();
    // Movement: |0.6-0.5|+|0.4-0.5| + |0.7-0.6|+|0.3-0.4| = 0.4; volume traded 20.
    // Epsilon 1e-12 covers the four f64 subtractions of decimal fractions.
    assert_relative_eq!(efficiency(&refs), 0.4 / 20.0, epsilon = 1e-12);
    // No volume traded across the window -> 0, never inf.
    let flat = [
        MarketSignal::new(id, vec![0.5, 0.5], 0, 10.0),
        MarketSignal::new(id, vec![0.6, 0.4], 0, 10.0),
    ];
    let refs: Vec<&MarketSignal> = flat.iter().collect();
    assert_eq!(efficiency(&refs), 0.0);
}

#[test]
fn signal_series_with_zero_capacity_still_keeps_latest() {
    let mut series = SignalSeries::new(0);
    series.push(MarketSignal::new(Uuid::new_v4(), vec![0.5, 0.5], 7, 1.0));
    assert_eq!(series.latest().map(|s| s.trade_count), Some(7));
}

// ---------------------------------------------------------------------------
// Trading lifecycle
// ---------------------------------------------------------------------------

#[test]
fn dust_position_can_be_fully_closed() {
    // 0.3 - 0.1 - 0.1 = 0.09999999999999998 < 0.1 in f64; a trader must
    // still be able to sell their "last 0.1" and end flat.
    let trader = Uuid::new_v4();
    let mut portfolio = Portfolio::new(trader);
    portfolio.add_cash(100.0).unwrap();
    let mut market = binary_market(100.0);
    execute_trade(&mut market, trader, 0, 0.3, &mut portfolio).unwrap();
    for _ in 0..3 {
        execute_trade(&mut market, trader, 0, -0.1, &mut portfolio).unwrap();
    }
    assert!(portfolio.position(market.id(), 0).is_none());
    assert_eq!(portfolio.shares(market.id(), 0), 0.0);
    // The market's outstanding shares are back to zero as well.
    assert!(market.quantities()[0].abs() <= 1e-12);
}

#[test]
fn over_sell_beyond_tolerance_is_refused() {
    let trader = Uuid::new_v4();
    let mut portfolio = Portfolio::new(trader);
    portfolio.add_cash(100.0).unwrap();
    let mut market = binary_market(100.0);
    execute_trade(&mut market, trader, 0, 1.0, &mut portfolio).unwrap();
    let err = execute_trade(&mut market, trader, 0, -1.001, &mut portfolio).unwrap_err();
    assert!(matches!(err, TradeError::InsufficientPosition { .. }));
    assert_eq!(portfolio.shares(market.id(), 0), 1.0);
}

#[test]
fn trade_on_closed_market_is_a_market_closed_error() {
    let trader = Uuid::new_v4();
    let mut portfolio = Portfolio::new(trader);
    portfolio.add_cash(100.0).unwrap();
    let mut market = binary_market(100.0);
    market.close().unwrap();
    let err = execute_trade(&mut market, trader, 0, 1.0, &mut portfolio).unwrap_err();
    assert!(matches!(err, TradeError::MarketClosed), "got {err:?}");
}

#[test]
fn trade_after_resolve_is_refused() {
    let trader = Uuid::new_v4();
    let mut portfolio = Portfolio::new(trader);
    portfolio.add_cash(100.0).unwrap();
    let mut market = binary_market(100.0);
    market.resolve(1).unwrap();
    assert!(execute_trade(&mut market, trader, 0, 1.0, &mut portfolio).is_err());
    assert!(matches!(market.trade_cost(0, 1.0), Err(MarketError::MarketClosed)));
}

#[test]
fn resolve_is_idempotent_for_the_same_outcome() {
    let mut market = binary_market(100.0);
    market.resolve(1).unwrap();
    // Same outcome again: no-op success (safe to retry).
    market.resolve(1).unwrap();
    assert_eq!(market.resolved_outcome(), Some(1));
    // A different outcome is a contradiction and must be refused.
    assert!(matches!(market.resolve(0), Err(MarketError::MarketResolved)));
    assert_eq!(market.resolved_outcome(), Some(1));
    assert_eq!(market.state(), MarketState::Resolved);
}

#[test]
fn resolve_with_no_trades_and_out_of_bounds() {
    let mut market = binary_market(100.0);
    assert!(matches!(
        market.resolve(2),
        Err(MarketError::InvalidOutcome { outcome: 2, num_outcomes: 2 })
    ));
    assert_eq!(market.state(), MarketState::Open);
    market.resolve(0).unwrap();
    let portfolio = Portfolio::new(Uuid::new_v4());
    assert_eq!(portfolio.market_payout(&market, 0), 0.0);
}

#[test]
fn execute_trade_refuses_zero_and_non_finite_amounts() {
    let trader = Uuid::new_v4();
    let mut portfolio = Portfolio::new(trader);
    portfolio.add_cash(100.0).unwrap();
    let mut market = binary_market(100.0);
    assert!(matches!(
        execute_trade(&mut market, trader, 0, 0.0, &mut portfolio),
        Err(TradeError::ZeroAmount)
    ));
    assert!(matches!(
        execute_trade(&mut market, trader, 0, -0.0, &mut portfolio),
        Err(TradeError::ZeroAmount)
    ));
    for amount in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let err = execute_trade(&mut market, trader, 0, amount, &mut portfolio).unwrap_err();
        assert!(matches!(err, TradeError::InvalidAmount(_)), "amount {amount}: {err:?}");
    }
    // Nothing leaked into state.
    assert_eq!(market.trade_stats(), (0, 0.0));
    assert_eq!(portfolio.cash(), 100.0);
}

// ---------------------------------------------------------------------------
// Serde: a deserialized market must satisfy the same invariants as a new one
// ---------------------------------------------------------------------------

#[test]
fn deserialized_market_is_validated() {
    let market = binary_market(100.0);
    let mut v = serde_json::to_value(&market).unwrap();

    v["liquidity"] = serde_json::json!(0.0);
    assert!(serde_json::from_value::<Market>(v.clone()).is_err(), "b = 0 accepted");

    v["liquidity"] = serde_json::json!(100.0);
    v["quantities"] = serde_json::json!([0.0]);
    assert!(
        serde_json::from_value::<Market>(v.clone()).is_err(),
        "quantities/outcomes length mismatch accepted"
    );

    v["quantities"] = serde_json::json!([0.0, -1.0]);
    assert!(serde_json::from_value::<Market>(v.clone()).is_err(), "negative q accepted");

    v["quantities"] = serde_json::json!([0.0, 0.0]);
    v["outcomes"] = serde_json::json!(["A", "a"]);
    assert!(serde_json::from_value::<Market>(v.clone()).is_err(), "duplicate labels accepted");

    v["outcomes"] = serde_json::json!(["A", "B"]);
    v["resolved_outcome"] = serde_json::json!(5);
    v["state"] = serde_json::json!("Resolved");
    assert!(
        serde_json::from_value::<Market>(v.clone()).is_err(),
        "resolved outcome out of bounds accepted"
    );

    v["resolved_outcome"] = serde_json::json!(1);
    let ok: Market = serde_json::from_value(v).unwrap();
    assert_eq!(ok.resolved_outcome(), Some(1));
}

#[test]
fn deserialized_portfolio_is_validated() {
    let mut portfolio = Portfolio::new(Uuid::new_v4());
    portfolio.add_cash(10.0).unwrap();
    let mut v = serde_json::to_value(&portfolio).unwrap();
    v["cash"] = serde_json::json!(-1.0);
    assert!(serde_json::from_value::<Portfolio>(v).is_err(), "negative cash accepted");
}

// ---------------------------------------------------------------------------
// Engine end-to-end (mirrors README)
// ---------------------------------------------------------------------------

#[test]
fn engine_can_fund_a_trader_through_its_public_api() {
    let mut engine = Engine::new();
    let market_id = engine
        .create_market("Rain?".into(), vec!["Yes".into(), "No".into()], 100.0)
        .unwrap();
    let trader = Uuid::new_v4();
    engine.deposit(trader, 50.0).unwrap();
    let trade = engine.trade(market_id, trader, 0, 10.0).unwrap();
    assert!(trade.cost > 0.0);
    let portfolio = engine.portfolio(trader).unwrap();
    // Epsilon 1e-12: one subtraction of two numbers of magnitude ~50.
    assert_relative_eq!(portfolio.cash(), 50.0 - trade.cost, epsilon = 1e-12);
    assert!(matches!(
        engine.deposit(trader, -1.0),
        Err(TradeError::InvalidAmount(_))
    ));
    assert!(matches!(
        engine.deposit(trader, f64::NAN),
        Err(TradeError::InvalidAmount(_))
    ));
}

#[test]
fn engine_settle_pays_winners_once_and_is_bounded() {
    let mut engine = Engine::new();
    let b = 100.0;
    let market_id = engine
        .create_market("Race".into(), vec!["A".into(), "B".into(), "C".into()], b)
        .unwrap();
    let alice = Uuid::new_v4();
    let bob = Uuid::new_v4();
    engine.deposit(alice, 1000.0).unwrap();
    engine.deposit(bob, 1000.0).unwrap();
    engine.trade(market_id, alice, 0, 40.0).unwrap();
    engine.trade(market_id, bob, 1, 25.0).unwrap();
    engine.trade(market_id, alice, 1, 5.0).unwrap();

    // Settle before resolve is refused.
    assert!(matches!(engine.settle(market_id), Err(MarketError::MarketNotResolved)));

    let pool = engine.market(market_id).unwrap().pool();
    engine.resolve(market_id, 1).unwrap();
    let payouts = engine.settle(market_id).unwrap();
    let total: f64 = payouts.iter().map(|(_, amount)| amount).sum();
    // Winning shares: bob 25 + alice 5 = 30, one unit each.
    assert_relative_eq!(total, 30.0, epsilon = 1e-12);
    // Worst-case market-maker loss is b·ln(n).
    assert!(total - pool <= b * 3.0_f64.ln() + 1e-9);

    // Idempotent: a second settlement pays nothing and changes nothing.
    let alice_cash = engine.portfolio(alice).unwrap().cash();
    let again = engine.settle(market_id).unwrap();
    assert!(again.is_empty());
    assert_eq!(engine.portfolio(alice).unwrap().cash(), alice_cash);
    assert!(engine.portfolio(alice).unwrap().position(market_id, 1).is_none());
    assert!(engine.portfolio(alice).unwrap().position(market_id, 0).is_none());
}

#[test]
fn engine_settle_on_unknown_market() {
    let mut engine = Engine::new();
    assert!(matches!(
        engine.settle(Uuid::new_v4()),
        Err(MarketError::MarketNotFound)
    ));
}
