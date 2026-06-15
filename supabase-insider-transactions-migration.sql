-- Open-market insider transactions (SEC Form 4, codes P/S only). Idempotent.
create table if not exists insider_transactions (
  txn_uid     text primary key,                 -- sha1(accession|owner|date|code|shares|price)
  ticker      text not null,
  owner       text,
  role        text,
  txn_date    date,
  code        text,                              -- P (open-market buy) | S (open-market sell)
  side        text,                              -- BUY | SELL
  shares       numeric,
  price        numeric,                          -- price/share transacted
  value_usd    numeric,                          -- shares * price
  scheduled    boolean default false,            -- 10b5-1 pre-scheduled plan (sell = noise)
  comp_related boolean default false,            -- exercise/grant/conversion in same filing
  accession   text,
  filing_date date,
  created_at  timestamptz not null default now()
);

create index if not exists insider_txn_ticker_date_idx on insider_transactions (ticker, txn_date desc);
create index if not exists insider_txn_date_idx on insider_transactions (txn_date desc);
create index if not exists insider_txn_accession_idx on insider_transactions (accession);

-- Public SEC data → public SELECT; writes only via service role (the refresh job).
alter table insider_transactions enable row level security;
do $$ begin
  create policy insider_txn_public_read on insider_transactions for select using (true);
exception when duplicate_object then null; end $$;
