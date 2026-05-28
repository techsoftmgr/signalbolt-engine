# SignalBolt — Signal Flow Architecture

**Last updated:** 2026-05-28

This document explains how a signal travels from raw market data to a fired
signal on the user's phone, through live management, to daily validation.

---

## Core principle: event-driven, not schedule-polled

The engine reacts to the **live Alpaca WebSocket stream**. It does NOT poll on a
clock as its primary mechanism. The APScheduler interval jobs exist only as a
**backstop** in case a stream event is missed.

Two kinds of stream events drive everything:

- **Trade ticks** (`on_trade`) — every executed trade. High frequency (kHz on
  liquid names). Drives: live prices, trade tape, per-tick compression
  breakout firing, tick-momentum scans, and real-time SL/TP management.
- **Bar closes** (`on_bar`) — when a 1m/5m/15m/1h bar finalizes, Alpaca PUSHES
  it over the WebSocket. This is an *event*, not a clock poll. Drives the
  strategy scans (SMC, pullback, compression staging).

---

## Architecture diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         ALPACA SIP WebSocket                               │
│                    (one connection, worker process)                        │
└───────────────┬──────────────────────────────────┬───────────────────────┘
                │ trade ticks (kHz)                 │ bar closes (1m/5m/15m/1h)
                ▼                                    ▼
   ┌────────────────────────────┐      ┌──────────────────────────────────┐
   │  on_trade (stream.py)      │      │  on_bar (stream.py)              │
   │  ─ price_store (live px)   │      │  ─ triggers run_strategy_by_type │
   │  ─ trade_tape (blocks/VWAP)│      │    on the bar-close boundary     │
   │  ─ per-tick compression ───┼──┐   └───────────────┬──────────────────┘
   │  ─ tick-momentum scan      │  │                   │
   │  ─ real-time SL/TP check   │  │                   ▼
   └────────────────────────────┘  │   ┌──────────────────────────────────┐
                                    │   │  run_strategy_by_type (runner)   │
   ┌────────────────────────────┐  │   │  per ticker (~150):              │
   │ APScheduler BACKSTOP        │  │   │   1. chop check                  │
   │ (10-min day / 5-min scalp)  │──┼──▶│   2. manipulation check          │
   │ only if stream event missed │  │   │   3. THREE DETECTORS:            │
   └────────────────────────────┘  │   │      • SMC pipeline              │
                                    │   │      • Pullback detector         │
                                    │   │      • Compression: STAGE zone   │
                                    │   │        + fire if already broke   │
                                    │   └───────────────┬──────────────────┘
                                    │                   │ setup found (any detector)
                                    │                   ▼
   per-tick breakout ──────────────┘   ┌──────────────────────────────────┐
   fires here when price                │  ENTRY GATE STACK (entry_gate.py)│
   crosses staged envelope ────────────▶│   1. 15m trend                   │
                                         │   2. 5m MACD     [skip: swing]   │
                                         │   3. 1m reversal [skip: swing]   │
                                         │   4. patterns                    │
                                         │   5. spread                      │
                                         │   6. tape        [skip: swing]   │
                                         └───────┬──────────────┬───────────┘
                                          ALL pass         ANY fail
                                                 │                │
                                                 ▼                ▼
                              ┌──────────────────────┐   ┌─────────────────────┐
                              │ FIRE PIPELINE         │   │ entry_gate_         │
                              │  • sl_tp_engine       │   │ rejections table    │
                              │  • risk_manager       │   │ (validated daily)   │
                              │  • tape_bonus         │   └─────────────────────┘
                              │  • _write_signal      │
                              │     - tag detector    │
                              │     - subscribe ticker│
                              │     - push to phone   │
                              └──────────┬────────────┘
                                         ▼
                              ┌──────────────────────┐
                              │  signals table        │
                              │  → app Signals tab     │
                              └──────────┬────────────┘
                                         │ while active
                                         ▼
                  ┌────────────────────────────────────────────┐
                  │  LIVE MANAGEMENT                            │
                  │  every tick:  _check_rt_levels             │
                  │     T2 → close win · T1 → SL to breakeven  │
                  │     SL → close loss · near-stop → warn     │
                  │  every 5m:   signal_monitor (RSI/EOD/expiry)│
                  │  every 15m:  tracker (reconcile W/L)        │
                  └────────────────────┬───────────────────────┘
                                       ▼
                  ┌────────────────────────────────────────────┐
                  │  VALIDATION (2:30 PM CDT daily)            │
                  │  gate_validator: replay rejections,        │
                  │  backfill would_have_won + pnl,            │
                  │  → Gate Effectiveness card                 │
                  └────────────────────────────────────────────┘
```

---

## Detailed flow — worked example

**Scenario:** NVDA consolidates tightly between $211.50–$212.00 from 9:00–9:15
AM CDT, then breaks out at 9:23:47 AM CDT.

### T+0 — 9:15:00 AM CDT — 15m bar closes
- Alpaca pushes the closed 15m bar over the WebSocket → `on_bar` fires
- `run_strategy_by_type("day_trade")` runs
- For NVDA: SMC finds no clean structure (still consolidating)
- `compression_detector.detect_zone(df)` sees 4 tight bars (range $0.50 < 0.55×ATR)
- → `stream.stage_compression_zone("NVDA", high=212.00, low=211.50, atr=...)`
- NVDA is now **staged** in the per-tick watch set. No signal yet.

### T+8m 47s — 9:23:47 AM CDT — breakout tick
- A trade prints NVDA @ $212.25 → `on_trade` fires
- `_check_compression_breakout("NVDA", 212.25)` runs
- $212.25 ≥ $212.00 × 1.001 ($212.21 upper buffer) → **LONG breakout**
- Zone removed (one fire per staging), `fire_compression_breakout` dispatched
  on the scan executor (doesn't block the event loop)

### T+8m 48s — fire pipeline
- `_has_active_signal` → none, continue
- `sl_tp_engine.calculate` → SL $210.80, T1 $213.90, T2 $215.40, R:R 1.6
- **Entry gate stack:**
  - 15m trend: EMA9 > EMA21 ✅
  - 5m MACD: histogram positive ✅
  - 1m reversal: last close > prev ✅
  - patterns: not overextended, volume OK ✅
  - spread: 0.04% ✅
  - tape: 6.9 trades/sec, block prints present ✅
  - **ALL PASS**
- `risk_manager`: portfolio OK ✅
- `tape_bonus`: +5 (institutional blocks present) → confidence 75 → 80
- `_write_signal`:
  - INSERT into signals, `detector_source: COMPRESSION`, `fire_path: per_tick`
  - subscribe NVDA to live trade stream (real-time SL/TP from now)
  - push notification → phone

### T+8m 48s — appears on phone
- Signals tab shows NVDA LONG @ $212.25, **COMP** badge, 80 confidence
- Live price ticks in real-time on the card

### Live management
- 9:31 AM CDT — NVDA hits T1 $213.90 → stop auto-moves to $212.25 (breakeven),
  card flips to "Stop (B/E)", rides toward T2
- 9:52 AM CDT — NVDA hits T2 $215.40 → close as WIN +1.48%, inline insight
  updates within ~1 sec

### 2:30 PM CDT — validation
- Any rejections that day get replayed
- NVDA fired, so it's in the fired-signal population, not rejections
- Its outcome contributes to the COMPRESSION detector's win-rate tally

---

## The three detectors compared

| Detector | When it fires | Entry timing | Trade-off |
|---|---|---|---|
| **SMC** | After BOS/CHoCH/FVG structure confirms | Late (50-70% into move) | Higher WR per signal, misses early move |
| **Compression** | Per-tick, the instant price breaks the consolidation envelope | Earliest (at the breakout) | Catches the start, more fakeouts |
| **Pullback** | When price reclaims swing high/low after a pullback | Mid (after healthy retrace) | Good R:R, waits for confirmation |

All three feed the **same gate stack** and **same risk engine**. Only the
*timing* differs. The daily validator + closed-signal WR per `detector_source`
tells us which timing wins over time.

---

## What is genuinely real-time vs not

| Action | Real-time? |
|---|---|
| Live price on cards | ✅ Every tick |
| Trade tape / block prints | ✅ Every tick |
| Compression breakout FIRE | ✅ Every tick (the breakout tick) |
| SL / TP / breakeven management | ✅ Every tick |
| Tick-momentum scan (≥0.4% move) | ✅ ~1 sec after threshold |
| SMC / pullback signal fire | ⚡ On bar close (a stream event, not a clock) |
| RSI/momentum exit, EOD warnings | ⏱ Every 5 min (signal_monitor) |
| Win/loss reconciliation | ⏱ Every 15 min (tracker) |
| Gate validation | ⏱ Daily 2:30 PM CDT |

**Bottom line:** entries and exits are event-driven off the live stream.
Compression fires on the literal breakout tick. The only true schedules are
the backstop scans, the 5-min monitor, the 15-min tracker, and the daily
validator — none of which gate the speed of a real breakout entry.
