# DNSE trading-token minter

`refresh_token.py` is the **sole producer** of `workdir/state/dnse_trading_token.json`,
which the plugin only ever **reads**. A token is valid ~8h and self-invalidating (a new
OTP kills the previous one), so mint once each trading morning — the plugin picks up the
new file automatically on its next read.

## Daily check (run this at ~08:05, after the cron)

`token_status.py` tells you whether the 08:00 job left a **working** token — and lets you
fix it on the spot:

```bash
.venv/bin/python plugins/dnse/tools/token_status.py       # add --refresh to force one
```

It prints the mint time + age vs the 8h TTL, whether it was minted after 08:00 today, the
cron-log tail, and a **live probe** — a harmless cancel of a bogus order id: DNSE checks
the token header before it looks the order up, so `INVALID_TRADING_TOKEN` means the token
is bad while any other reply (not-found, session-closed, …) means it was accepted. If the
token isn't good it offers to refresh: it sends the OTP, you read your email and type the
code, it writes the new token.

## Manual (works today, no Gmail creds)

```bash
.venv/bin/python plugins/dnse/tools/refresh_token.py --send        # send the OTP email
.venv/bin/python plugins/dnse/tools/refresh_token.py --otp 123456  # read it, pass it in
```

## Auto (for the cron) — needs a Gmail app password

1. On the Google account: enable **2-Step Verification**, then create an **App password**
   (Google Account → Security → App passwords → "Mail").
2. Store the creds in a private, gitignored env file:

   ```bash
   umask 077
   cat > workdir/state/dnse_gmail.env <<'EOF'
   DNSE_GMAIL_USER=you@gmail.com
   DNSE_GMAIL_APP_PASSWORD=xxxx xxxx xxxx xxxx
   EOF
   ```

3. Install ONE daily cron at 08:00 ICT (TTL 8h covers the morning + afternoon session):

   ```cron
   CRON_TZ=Asia/Ho_Chi_Minh
   0 8 * * 1-5 cd /home/mike/workspace/github/pynecore && set -a && . workdir/state/dnse_gmail.env && .venv/bin/python plugins/dnse/tools/refresh_token.py >> workdir/state/refresh_token.log 2>&1
   ```

If a morning cron fails (no OTP arrived, Gmail hiccup), fall back to the manual leg above
— the plugin also re-reads the token file on an `INVALID_TRADING_TOKEN` reject, so a
mid-session manual refresh is picked up without a restart.

## Security

The state file and the env file are **order-placement authority**: written `0600`, kept
under `workdir/state/` (gitignored), never committed, never logged. Treat them like
`api_secret`.
