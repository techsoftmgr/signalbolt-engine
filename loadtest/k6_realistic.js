/**
 * k6 load test simulating SignalBolt's real user mix at 1000 concurrent users.
 *
 * What an actual user does:
 *   1. Opens the app → 1x /sync-subscription, 1x /signals, 1x /market/status
 *   2. Stays on Signals tab → keeps /ws/prices WebSocket open
 *   3. Switches tabs → polls /market/heatmap (15s), /quant/dashboard (60s),
 *      /news/reaction (5min), /signals/community (30s)
 *   4. /market/status banner polls every 60s
 *
 * This script models that mix. Two scenarios run in parallel:
 *   - "viewers"     — 850 users with WS open + light REST polling
 *   - "browsers"    — 150 users actively switching tabs / triggering loads
 *
 * Stages: ramps from 0 → 1000 over 2 min, holds 5 min, ramps down 1 min.
 * Total run time: ~8 min.
 *
 * Run with:
 *   k6 run --env BASE=https://signalbolt-engine.fly.dev \
 *          --env WS_BASE=wss://signalbolt-engine.fly.dev \
 *          --env JWT=eyJhbGc...  loadtest/k6_realistic.js
 *
 * The JWT must be a real Supabase user token (steal one from your app's
 * AsyncStorage during dev, or generate one with the supabase admin API).
 * Endpoints that require auth fail with 401 otherwise.
 */

import http from 'k6/http';
import ws from 'k6/ws';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

// ── Config (override via --env) ──────────────────────────────────────────────
const BASE    = __ENV.BASE    || 'https://signalbolt-engine.fly.dev';
const WS_BASE = __ENV.WS_BASE || BASE.replace(/^http/, 'ws');
const JWT     = __ENV.JWT     || '';  // Supabase access token

const TICKERS = ['SPY','QQQ','AAPL','NVDA','MSFT','TSLA','META','AMD','GOOGL','PLTR'];

// ── Custom metrics ───────────────────────────────────────────────────────────
const wsConnectErrors = new Rate('ws_connect_errors');
const wsMessageRate   = new Rate('ws_messages_received');
const apiErrors       = new Rate('api_errors');
const tabSwitchTime   = new Trend('tab_switch_total_ms');

// ── Scenarios ────────────────────────────────────────────────────────────────
export const options = {
  scenarios: {
    // 85% of users: sit on Signals tab with WebSocket open, occasional polls
    viewers: {
      executor: 'ramping-vus',
      exec:     'viewerSession',
      startVUs: 0,
      stages: [
        { duration: '2m', target: 850 },   // ramp to 850
        { duration: '5m', target: 850 },   // hold
        { duration: '1m', target: 0   },   // ramp down
      ],
      gracefulStop: '30s',
    },
    // 15% of users: actively browsing — switching tabs, triggering REST loads
    browsers: {
      executor: 'ramping-vus',
      exec:     'browserSession',
      startVUs: 0,
      stages: [
        { duration: '2m', target: 150 },
        { duration: '5m', target: 150 },
        { duration: '1m', target: 0   },
      ],
      gracefulStop: '30s',
    },
  },

  // Hard thresholds — test fails if these regress
  thresholds: {
    'http_req_duration{name:health}':        ['p(95)<200'],   // /health p95 < 200ms
    'http_req_duration{name:market_status}': ['p(95)<300'],
    'http_req_duration{name:signals}':       ['p(95)<2000'],
    'http_req_duration{name:heatmap}':       ['p(95)<3000'],
    'http_req_failed':                       ['rate<0.02'],   // <2% errors overall
    'ws_connect_errors':                     ['rate<0.05'],   // <5% WS failures
    'tab_switch_total_ms':                   ['p(95)<5000'],  // tab switch end-to-end <5s
  },

  // Don't dump 1k threads of debug noise
  summaryTrendStats: ['avg','min','med','max','p(90)','p(95)','p(99)'],
};

// ── Helpers ──────────────────────────────────────────────────────────────────
function authedGet(path, name) {
  const url = `${BASE}${path}`;
  const opts = {
    headers: JWT ? { 'Authorization': `Bearer ${JWT}` } : {},
    tags: { name },
  };
  const r = http.get(url, opts);
  apiErrors.add(r.status >= 400);
  return r;
}

function publicGet(path, name) {
  const r = http.get(`${BASE}${path}`, { tags: { name } });
  apiErrors.add(r.status >= 400);
  return r;
}

// ── Scenario 1: Viewer (most users — passive, WS-heavy) ──────────────────────
export function viewerSession() {
  // Initial load
  publicGet('/health',        'health');
  publicGet('/market/status', 'market_status');
  authedGet(`/signals?strategy_type=day_trade`, 'signals');

  // Open WebSocket for live prices — this is the LOAD-BEARING path
  const wsUrl = `${WS_BASE}/ws/prices`;
  const wsRes = ws.connect(wsUrl, null, function(socket) {
    socket.on('open', () => {
      socket.send(JSON.stringify({ subscribe: TICKERS }));
    });
    socket.on('message', () => {
      wsMessageRate.add(1);
    });
    // Keep the WS open for 4 minutes (typical user dwell time on Signals tab)
    socket.setTimeout(() => { socket.close(); }, 240_000);

    // While the WS is open, do the background tab-stays-focused polling
    socket.setInterval(() => {
      publicGet('/market/status', 'market_status');
    }, 60_000);
  });
  wsConnectErrors.add(wsRes && wsRes.status !== 101);
}

// ── Scenario 2: Browser (active — switches tabs, hits expensive endpoints) ──
export function browserSession() {
  // Boot
  publicGet('/market/status', 'market_status');
  authedGet(`/signals?strategy_type=day_trade`, 'signals');
  sleep(2);

  // Tab switch: Heatmap (fetches /market/heatmap, polls every 15s)
  const t0 = Date.now();
  authedGet('/market/heatmap', 'heatmap');
  tabSwitchTime.add(Date.now() - t0);
  sleep(15);
  authedGet('/market/heatmap', 'heatmap');
  sleep(15);

  // Tab switch: Quant (60s refresh)
  authedGet('/quant/dashboard', 'quant');
  sleep(30);

  // Tab switch: News (5min refresh, so just one fetch)
  authedGet('/news/reaction', 'news');
  sleep(30);

  // Tab switch: Community (30s refresh)
  authedGet('/signals/community', 'community');
  sleep(30);
  authedGet('/signals/community', 'community');
  sleep(30);

  // Back to Signals
  authedGet(`/signals?strategy_type=day_trade`, 'signals');
}
