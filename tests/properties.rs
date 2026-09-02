//! Property-based tests for the LMSR invariants.
//!
//! Strategy notes: liquidity is sampled log-uniformly so tiny and huge `b`
//! both get coverage; quantities are sampled in units of `b` because every
//! LMSR quantity only ever appears as `q / b`.

use approx::assert_relative_eq;
use ghostsignals::{
    cost, entropy, normalized_entropy, prices, trade_cost, Engine, Market, MarketSignal,
    Portfolio, SignalSeries, Trade,
};
use proptest::prelude::*;
use uuid::Uuid;

/// Log-uniform liquidity across ten orders of magnitude.
fn liquidity() -> impl Strategy<Value = f64> {
    (-2.0f64..8.0).prop_map(|e| 10f64.powf(e))
}

/// Quantities expressed in units of b, wide enough to saturate exp().
fn quantities(n: impl Into<proptest::collection::SizeRange>) -> impl Strategy<Value = Vec<f64>> {
    proptest::collection::vec(-2000.0f64..2000.0, n)
}

fn labels(n: usize) -> Vec<String> {
    (0..n).map(|i| format!("O{i}")).collect()
}

/// Sum of price rounding errors: n terms each within a few ulps of 1.
fn price_sum_eps(n: usize) -> f64 {
    1e-14 * n as f64
}

/// 512 cases by default; `PROPTEST_CASES=n` overrides for a heavier run.
fn config() -> ProptestConfig {
    let cases = std::env::var("PROPTEST_CASES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(512);
    ProptestConfig { cases, ..ProptestConfig::default() }
}

proptest! {
    #![proptest_config(config())]

    // -- prices --------------------------------------------------------------

    #[test]
    fn prices_are_a_distribution(units in quantities(2..12usize), b in liquidity()) {
        let q: Vec<f64> = units.iter().map(|u| u * b).collect();
        // A finite q/b is guaranteed by construction, so this must succeed.
        let p = prices(&q, b).unwrap();
        prop_assert_eq!(p.len(), q.len());
        for &x in &p {
            prop_assert!(x.is_finite());
            prop_assert!((0.0..=1.0).contains(&x), "price {} out of [0,1]", x);
        }
        let sum: f64 = p.iter().sum();
        prop_assert!((sum - 1.0).abs() <= price_sum_eps(p.len()), "sum = {}", sum);
        // Never exactly degenerate: the arg-max outcome always has p > 0.
        let (imax, _) = q.iter().enumerate().fold((0, f64::NEG_INFINITY), |acc, (i, &v)| {
            if v > acc.1 { (i, v) } else { acc }
        });
        prop_assert!(p[imax] > 0.0);
    }

    #[test]
    fn prices_are_monotone_in_quantity(
        units in quantities(2..8usize),
        b in liquidity(),
        i in 0usize..8,
        bump in 1e-6f64..50.0,
    ) {
        let i = i % units.len();
        let q: Vec<f64> = units.iter().map(|u| u * b).collect();
        let mut q2 = q.clone();
        q2[i] += bump * b;
        let p1 = prices(&q, b).unwrap();
        let p2 = prices(&q2, b).unwrap();
        prop_assert!(p2[i] >= p1[i], "p_i fell: {} -> {}", p1[i], p2[i]);
        for j in 0..q.len() {
            if j != i {
                // Tolerance: one ulp of renormalisation noise.
                prop_assert!(p2[j] <= p1[j] + 1e-15, "p_{} rose: {} -> {}", j, p1[j], p2[j]);
            }
        }
    }

    #[test]
    fn prices_are_translation_invariant(units in quantities(2..8usize), b in liquidity(), shift in -500.0f64..500.0) {
        let q: Vec<f64> = units.iter().map(|u| u * b).collect();
        let q2: Vec<f64> = q.iter().map(|x| x + shift * b).collect();
        let p1 = prices(&q, b).unwrap();
        let p2 = prices(&q2, b).unwrap();
        for (a, c) in p1.iter().zip(&p2) {
            // 1e-12 absolute: exp(x - lse) with x shifted by up to 500 costs
            // a few hundred ulps at most.
            prop_assert!((a - c).abs() <= 1e-12, "{} vs {}", a, c);
        }
    }

    // -- cost ----------------------------------------------------------------

    #[test]
    fn cost_is_finite_and_bounded_by_max_quantity(units in quantities(1..12usize), b in liquidity()) {
        let q: Vec<f64> = units.iter().map(|u| u * b).collect();
        let c = cost(&q, b).unwrap();
        prop_assert!(c.is_finite());
        let qmax = q.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let n = q.len() as f64;
        // max(q) <= C(q) <= max(q) + b ln n, up to rounding of magnitude |qmax|.
        let tol = 1e-12 * (qmax.abs() + b);
        prop_assert!(c >= qmax - tol, "C={} < max q={}", c, qmax);
        prop_assert!(c <= qmax + b * n.ln() + tol, "C={} > max q + b ln n = {}", c, qmax + b * n.ln());
    }

    #[test]
    fn trade_cost_agrees_with_cost_difference(
        units in quantities(2..8usize),
        b in liquidity(),
        i in 0usize..8,
        delta_units in -50.0f64..50.0,
    ) {
        let i = i % units.len();
        let q: Vec<f64> = units.iter().map(|u| u * b).collect();
        let delta = delta_units * b;
        let tc = trade_cost(&q, b, i, delta).unwrap();
        let mut q2 = q.clone();
        q2[i] += delta;
        let diff = cost(&q2, b).unwrap() - cost(&q, b).unwrap();
        // The direct difference carries cancellation error of ~ulp(C) which
        // is 1e-16·|C|; |C| <= max|q| + b ln n.
        let scale = q.iter().fold(b, |m, x| m.max(x.abs())) + delta.abs();
        prop_assert!((tc - diff).abs() <= 1e-9 * scale + 1e-12, "tc={} diff={}", tc, diff);
        // Sign: buying costs, selling pays.
        if delta > 0.0 { prop_assert!(tc >= 0.0); } else { prop_assert!(tc <= 0.0); }
        // |cost| <= |delta| (a share pays at most one unit).
        prop_assert!(tc.abs() <= delta.abs() * (1.0 + 1e-12) + 1e-12);
    }

    #[test]
    fn round_trip_never_creates_money(
        units in quantities(2..6usize),
        b in liquidity(),
        i in 0usize..6,
        amount_units in 1e-6f64..100.0,
    ) {
        let i = i % units.len();
        let q: Vec<f64> = units.iter().map(|u| u * b).collect();
        let amount = amount_units * b;
        let buy = trade_cost(&q, b, i, amount).unwrap();
        let mut q2 = q.clone();
        q2[i] += amount;
        let sell = trade_cost(&q2, b, i, -amount).unwrap();
        // Net for the trader: -buy - sell must be <= 0 up to rounding of
        // the two costs (each accurate to ~1e-15 relative).
        let net = -buy - sell;
        prop_assert!(net <= 1e-12 * buy.abs().max(b) , "trader netted {} on a round trip", net);
    }

    #[test]
    fn many_small_trades_equal_one_large_trade(
        units in quantities(2..6usize),
        b in liquidity(),
        i in 0usize..6,
        total_units in 1e-3f64..20.0,
        steps in 1usize..64,
    ) {
        let i = i % units.len();
        let q: Vec<f64> = units.iter().map(|u| u * b).collect();
        let total = total_units * b;
        let one = trade_cost(&q, b, i, total).unwrap();
        let step = total / steps as f64;
        let mut running = q.clone();
        let mut summed = 0.0;
        for _ in 0..steps {
            summed += trade_cost(&running, b, i, step).unwrap();
            running[i] += step;
        }
        // 64 steps each with ~1e-15 relative error on a value <= total.
        prop_assert!((summed - one).abs() <= 1e-11 * total.max(b) + 1e-12, "sum {} vs one {}", summed, one);
    }

    // -- entropy ---------------------------------------------------------------

    #[test]
    fn entropy_is_bounded_by_ln_n(units in quantities(1..16usize), b in liquidity()) {
        let q: Vec<f64> = units.iter().map(|u| u * b).collect();
        let p = prices(&q, b).unwrap();
        let e = entropy(&p);
        let n = p.len() as f64;
        prop_assert!(e >= -1e-15);
        prop_assert!(e <= n.ln() + 1e-12, "H={} > ln n={}", e, n.ln());
        let ne = normalized_entropy(&p);
        prop_assert!((-1e-15..=1.0 + 1e-12).contains(&ne));
    }

    // -- engine: conservation and bounded loss ----------------------------------

    #[test]
    fn trading_conserves_money_and_bounds_market_maker_loss(
        n in 2usize..6,
        b in (0.0f64..3.0).prop_map(|e| 10f64.powf(e)),
        ops in proptest::collection::vec((0usize..3, 0usize..6, 0.01f64..30.0, any::<bool>()), 1..60),
    ) {
        let mut engine = Engine::new();
        let market_id = engine.create_market("Q?".into(), labels(n), b).unwrap();
        let traders: Vec<Uuid> = (0..3).map(|_| Uuid::new_v4()).collect();
        let initial = 1_000_000.0;
        for t in &traders { engine.deposit(*t, initial).unwrap(); }

        let mut applied = 0;
        for (ti, oi, amount_units, is_buy) in ops {
            let trader = traders[ti];
            let outcome = oi % n;
            let amount = amount_units * b * if is_buy { 1.0 } else { -1.0 };
            if let Ok(trade) = engine.trade(market_id, trader, outcome, amount) {
                applied += 1;
                prop_assert!(trade.cost.is_finite());
                prop_assert!(trade.cost.abs() <= amount.abs() * (1.0 + 1e-12) + 1e-12);
            }
            let market = engine.market(market_id).unwrap();
            // Outstanding shares are never negative (no naked shorts).
            for &qv in market.quantities() { prop_assert!(qv >= -1e-9 * b, "q went negative: {}", qv); }
            // Balances never negative.
            for t in &traders {
                let cash = engine.portfolio(*t).unwrap().cash();
                prop_assert!(cash >= 0.0, "cash negative: {}", cash);
                for (_, pos) in engine.portfolio(*t).unwrap().market_positions(market_id).into_iter().flatten() {
                    prop_assert!(pos.shares > 0.0, "empty/negative position kept: {}", pos.shares);
                }
            }
        }

        let market = engine.market(market_id).unwrap();
        prop_assert_eq!(market.trade_stats().0, applied as u64);
        let pool = market.pool();
        let total_cash: f64 = traders.iter().map(|t| engine.portfolio(*t).unwrap().cash()).sum();
        // Conservation: sum of balances + pool == initial deposits.
        // Tolerance 1e-9 relative on 3e6: sixty additions at 1e-16 each.
        prop_assert!(
            (total_cash + pool - 3.0 * initial).abs() <= 1e-9 * 3.0 * initial,
            "conservation broken: cash {} + pool {} != {}", total_cash, pool, 3.0 * initial
        );

        // Bounded worst-case loss: whichever outcome wins, payout - pool <= b ln n.
        let ln_n = (n as f64).ln();
        for outcome in 0..n {
            let payout: f64 = traders.iter().map(|t| engine.portfolio(*t).unwrap().market_payout(market, outcome)).sum();
            // Payout of outcome i equals the shares outstanding q_i.
            prop_assert!((payout - market.quantities()[outcome]).abs() <= 1e-9 * (payout.abs() + b));
            prop_assert!(
                payout - pool <= b * ln_n + 1e-9 * (pool.abs() + b),
                "loss {} exceeds b ln n = {}", payout - pool, b * ln_n
            );
        }

        // Settle the last outcome and re-check conservation including payouts.
        let winner = n - 1;
        engine.resolve(market_id, winner).unwrap();
        let payouts = engine.settle(market_id).unwrap();
        let paid: f64 = payouts.iter().map(|(_, a)| a).sum();
        let after_cash: f64 = traders.iter().map(|t| engine.portfolio(*t).unwrap().cash()).sum();
        prop_assert!((after_cash - (total_cash + paid)).abs() <= 1e-9 * 3.0 * initial);
        prop_assert!(pool - paid >= -b * ln_n - 1e-9 * (pool.abs() + b), "maker lost more than b ln n");
        // Trading after settlement is refused; settling again pays nothing.
        prop_assert!(engine.trade(market_id, traders[0], 0, 1.0).is_err());
        prop_assert!(engine.settle(market_id).unwrap().is_empty());
    }

    // -- serde round trips --------------------------------------------------------
    // Exact f64 equality after JSON needs serde_json's `float_roundtrip`
    // feature (its default parser is best-effort and can drift by an ulp).

    #[test]
    fn engine_round_trips_through_serde(
        n in 2usize..5,
        b in (0.0f64..3.0).prop_map(|e| 10f64.powf(e)),
        ops in proptest::collection::vec((0usize..2, 0usize..5, 0.1f64..10.0), 0..12),
    ) {
        let mut engine = Engine::new();
        let market_id = engine.create_market("Q?".into(), labels(n), b).unwrap();
        let traders = [Uuid::new_v4(), Uuid::new_v4()];
        for t in &traders { engine.deposit(*t, 1000.0).unwrap(); }
        for (ti, oi, amount) in ops {
            let _ = engine.trade(market_id, traders[ti], oi % n, amount * b);
        }
        let json = serde_json::to_string(&engine).unwrap();
        let back: Engine = serde_json::from_str(&json).unwrap();

        let m1 = engine.market(market_id).unwrap();
        let m2 = back.market(market_id).unwrap();
        prop_assert_eq!(m1.id(), m2.id());
        prop_assert_eq!(m1.quantities(), m2.quantities());
        prop_assert_eq!(m1.prices(), m2.prices());
        prop_assert_eq!(m1.pool(), m2.pool());
        prop_assert_eq!(m1.trade_stats(), m2.trade_stats());
        prop_assert_eq!(m1.state(), m2.state());
        prop_assert_eq!(m1.created_at(), m2.created_at());
        for t in &traders {
            let p1 = engine.portfolio(*t).unwrap();
            let p2 = back.portfolio(*t).unwrap();
            prop_assert_eq!(p1.cash(), p2.cash());
            prop_assert_eq!(p1.trades().len(), p2.trades().len());
            for o in 0..n {
                prop_assert_eq!(p1.position(market_id, o), p2.position(market_id, o));
            }
            for (a, c) in p1.trades().iter().zip(p2.trades()) {
                prop_assert_eq!(a.id, c.id);
                prop_assert_eq!(a.cost, c.cost);
                prop_assert_eq!(a.amount, c.amount);
                prop_assert_eq!(a.timestamp, c.timestamp);
            }
        }
        // Trade ids are unique across the whole engine.
        let mut ids: Vec<Uuid> = traders.iter().flat_map(|t| engine.portfolio(*t).unwrap().trades().iter().map(|tr| tr.id)).collect();
        let before = ids.len();
        ids.sort();
        ids.dedup();
        prop_assert_eq!(ids.len(), before);
    }

    #[test]
    fn value_types_round_trip_through_serde(
        prices_in in proptest::collection::vec(0.0f64..1.0, 1..6),
        trade_count in any::<u64>(),
        volume in 0.0f64..1e6,
        cap in 1usize..10,
    ) {
        let id = Uuid::new_v4();
        let signal = MarketSignal::new(id, prices_in.clone(), trade_count, volume);
        let s2: MarketSignal = serde_json::from_str(&serde_json::to_string(&signal).unwrap()).unwrap();
        prop_assert_eq!(s2.market_id, id);
        prop_assert_eq!(&s2.prices, &prices_in);
        prop_assert_eq!(s2.trade_count, trade_count);
        prop_assert_eq!(s2.volume, volume);
        prop_assert_eq!(s2.timestamp, signal.timestamp);

        let mut series = SignalSeries::new(cap);
        for _ in 0..(cap + 2) { series.push(signal.clone()); }
        let series2: SignalSeries = serde_json::from_str(&serde_json::to_string(&series).unwrap()).unwrap();
        prop_assert_eq!(series2.signals().len(), cap);
        prop_assert_eq!(series2.latest().map(|s| s.trade_count), Some(trade_count));

        let trade = Trade::new(id, Uuid::new_v4(), 1, 2.5, 1.25);
        let t2: Trade = serde_json::from_str(&serde_json::to_string(&trade).unwrap()).unwrap();
        prop_assert_eq!(t2.id, trade.id);
        prop_assert_eq!(t2.trader_id, trade.trader_id);
        prop_assert_eq!(t2.outcome, 1);
        prop_assert_eq!(t2.timestamp, trade.timestamp);

        let market = Market::new("Q?".into(), labels(3), 42.0).unwrap();
        let m2: Market = serde_json::from_str(&serde_json::to_string(&market).unwrap()).unwrap();
        prop_assert_eq!(m2.question(), market.question());
        prop_assert_eq!(m2.outcomes(), market.outcomes());
        prop_assert_eq!(m2.liquidity(), market.liquidity());

        let portfolio = Portfolio::new(id);
        let p2: Portfolio = serde_json::from_str(&serde_json::to_string(&portfolio).unwrap()).unwrap();
        prop_assert_eq!(p2.trader_id, id);
        prop_assert_eq!(p2.cash(), 0.0);
    }
}

#[test]
fn market_ids_are_unique() {
    let mut engine = Engine::new();
    let mut ids: Vec<Uuid> = (0..200)
        .map(|_| engine.create_market("Q?".into(), labels(2), 10.0).unwrap())
        .collect();
    ids.sort();
    ids.dedup();
    assert_eq!(ids.len(), 200);
}

#[test]
fn trade_history_is_in_execution_order() {
    let mut engine = Engine::new();
    let market_id = engine.create_market("Q?".into(), labels(2), 10.0).unwrap();
    let trader = Uuid::new_v4();
    engine.deposit(trader, 1000.0).unwrap();
    let mut executed = Vec::new();
    for i in 0..10 {
        executed.push(engine.trade(market_id, trader, i % 2, 1.0).unwrap().id);
    }
    let recorded: Vec<Uuid> = engine.portfolio(trader).unwrap().trades().iter().map(|t| t.id).collect();
    assert_eq!(recorded, executed);
    let ts: Vec<_> = engine.portfolio(trader).unwrap().trades().iter().map(|t| t.timestamp).collect();
    assert!(ts.windows(2).all(|w| w[0] <= w[1]), "timestamps not monotone: {ts:?}");
    let p = prices(engine.market(market_id).unwrap().quantities(), 10.0).unwrap();
    assert_relative_eq!(p[0], 0.5, epsilon = 1e-12);
}
