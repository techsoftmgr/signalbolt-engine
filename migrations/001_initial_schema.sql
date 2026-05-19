-- =============================================
-- SignalBolt Supabase Schema
-- Run this in the Supabase SQL Editor
-- =============================================

-- Profiles table (extends auth.users)
create table profiles (
  id uuid references auth.users on delete cascade,
  email text,
  tier text default 'free',
  created_at timestamp with time zone default timezone('utc'::text, now()),
  primary key (id)
);

-- Signals table
create table signals (
  id uuid default gen_random_uuid() primary key,
  ticker text not null,
  direction text not null check (direction in ('LONG', 'SHORT')),
  entry_price decimal,
  stop_loss decimal,
  target_one decimal,
  target_two decimal,
  confidence_score integer check (confidence_score between 0 and 100),
  ai_explanation text,
  timeframe text,
  status text default 'active' check (status in ('active', 'closed', 'cancelled')),
  created_at timestamp with time zone default timezone('utc'::text, now())
);

-- Subscriptions table
create table subscriptions (
  id uuid default gen_random_uuid() primary key,
  user_id uuid references profiles(id),
  stripe_customer_id text,
  plan text,
  status text,
  created_at timestamp with time zone default timezone('utc'::text, now())
);

-- Auto-create profile on signup
create or replace function public.handle_new_user()
returns trigger as $$
begin
  insert into public.profiles (id, email)
  values (new.id, new.email);
  return new;
end;
$$ language plpgsql security definer;

create trigger on_auth_user_created
  after insert on auth.users
  for each row execute procedure public.handle_new_user();

-- Enable realtime on signals table
alter publication supabase_realtime add table signals;

-- Row level security
alter table profiles enable row level security;
alter table signals enable row level security;
alter table subscriptions enable row level security;

-- Policies: signals viewable by all authenticated users
create policy "Authenticated users can view signals"
  on signals for select
  using (auth.role() = 'authenticated');

-- Policies: profiles
create policy "Users can view own profile"
  on profiles for select
  using (auth.uid() = id);

create policy "Users can update own profile"
  on profiles for update
  using (auth.uid() = id);

-- Policies: subscriptions
create policy "Users can view own subscription"
  on subscriptions for select
  using (auth.uid() = user_id);

-- Indexes for performance
create index signals_created_at_idx on signals (created_at desc);
create index signals_ticker_idx on signals (ticker);
create index signals_status_idx on signals (status);

-- Seed some sample signals for testing
insert into signals (ticker, direction, entry_price, stop_loss, target_one, target_two, confidence_score, ai_explanation, timeframe, status)
values
  ('AAPL', 'LONG', 189.50, 185.00, 195.00, 202.00, 82, 'Strong bullish momentum with RSI breakout above 60. Volume surge confirms institutional accumulation. MACD crossover on daily chart. Price consolidating above key 50-day MA.', '1D', 'active'),
  ('TSLA', 'SHORT', 245.80, 252.00, 235.00, 228.00, 74, 'Bearish divergence on RSI with price making new highs. Dark cloud cover candlestick pattern. Negative news catalyst from delivery numbers miss. Strong resistance at $250.', '4H', 'active'),
  ('NVDA', 'LONG', 875.00, 855.00, 910.00, 945.00, 91, 'AI infrastructure spending accelerating. Cup and handle breakout on weekly chart. Institutional buying detected through dark pool prints. Earnings revision cycle turning positive.', '1W', 'active'),
  ('SPY', 'SHORT', 521.50, 526.00, 512.00, 505.00, 68, 'Overbought conditions with VIX suppression. Distribution pattern forming on high volume. Seasonal headwinds in May. Options market pricing increased downside risk.', '1D', 'active'),
  ('BTC-USD', 'LONG', 62400.00, 59000.00, 68000.00, 75000.00, 79, 'Bitcoin halving supply shock impact materializing. On-chain metrics show accumulation by long-term holders. Technical breakout from 6-month consolidation range. Spot ETF inflows accelerating.', '1W', 'active');
