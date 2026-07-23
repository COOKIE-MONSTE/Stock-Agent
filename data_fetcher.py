import yfinance as yf
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import datetime
import json
from email.utils import parsedate_to_datetime
import os
import requests
import pandas as pd
import matplotlib
# Use the non-interactive Agg backend to avoid GUI window popup issues on Windows/servers
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import textwrap

def _format_pub_date(raw):
    """
    Normalizes a published-date value into a short, consistent display
    string like 'Jun 16, 2026'. Handles the formats we actually see:
      - Google News RSS pubDate (RFC 822, e.g. 'Mon, 16 Jun 2026 14:23:00 GMT')
      - yfinance's current news format (ISO 8601, e.g. '2026-06-16T14:23:00Z')
      - yfinance's legacy news format (unix timestamp int)
    Falls back to a truncated raw string if none of those parse.
    """
    if not raw:
        return "N/A"
    if isinstance(raw, (int, float)):
        try:
            return datetime.datetime.fromtimestamp(raw, tz=datetime.timezone.utc).strftime("%b %d, %Y")
        except (ValueError, OSError):
            return "N/A"
    raw = str(raw)
    for parse in (parsedate_to_datetime, lambda s: datetime.datetime.fromisoformat(s.replace("Z", "+00:00"))):
        try:
            return parse(raw).strftime("%b %d, %Y")
        except (ValueError, TypeError):
            continue
    return raw[:16]

def _parse_pub_datetime(raw):
    """
    Like _format_pub_date, but returns an actual timezone-aware datetime
    (or None) so callers can filter by recency. _format_pub_date is still
    used for the final display string; this is the machine-readable version.
    """
    if not raw:
        return None
    if isinstance(raw, (int, float)):
        try:
            return datetime.datetime.fromtimestamp(raw, tz=datetime.timezone.utc)
        except (ValueError, OSError):
            return None
    raw = str(raw)
    for parse in (parsedate_to_datetime,
                  lambda s: datetime.datetime.fromisoformat(s.replace("Z", "+00:00"))):
        try:
            dt = parse(raw)
            # Normalize naive datetimes to UTC so comparisons never crash.
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=datetime.timezone.utc)
            return dt
        except (ValueError, TypeError):
            continue
    return None


# Source-quality tiers. Higher score = more trustworthy / less spammy.
# Matched case-insensitively as substrings against the publisher string that
# Google News appends to each headline (e.g. "... - Reuters"). Anything not
# listed gets NEUTRAL_SOURCE_SCORE. Add/adjust names freely as you spot
# offenders in real output.
SOURCE_SCORES = {
    # Tier 1: primary financial news wires / papers of record
    "reuters": 10, "bloomberg": 10, "wall street journal": 10, "wsj": 10,
    "financial times": 10, "ft.com": 10, "the associated press": 9, "ap news": 9,
    # Tier 2: solid business press
    "cnbc": 7, "barron": 7, "marketwatch": 6, "forbes": 6, "the economist": 8,
    "business insider": 5, "yahoo finance": 5, "investor's business daily": 6,
    "seeking alpha": 4, "benzinga": 4,
    # Company primary sources (press releases / IR wires) — high signal, low spin
    "business wire": 8, "globe newswire": 8, "pr newswire": 8,
    # Tier 3: known aggregators / content mills — actively downweighted
    "insider monkey": 1, "simply wall st": 1, "zacks": 2, "the motley fool": 2,
    "motley fool": 2, "tipranks": 2, "gurufocus": 2, "stocktwits": 1,
    "24/7 wall st": 1, "invezz": 1, "247wallst": 1,
}
NEUTRAL_SOURCE_SCORE = 3


def _score_source(publisher):
    """Return a quality score for a publisher name (case-insensitive substring match)."""
    if not publisher:
        return NEUTRAL_SOURCE_SCORE
    p = publisher.lower()
    for name, score in SOURCE_SCORES.items():
        if name in p:
            return score
    return NEUTRAL_SOURCE_SCORE


def _get_fx_rate(from_currency, to_currency):
    """
    Fetches a spot FX rate to convert an amount in `from_currency` into `to_currency`,
    using yfinance currency-pair tickers (e.g. 'GBPUSD=X').
    Returns 1.0 if no conversion is needed, or None if a rate can't be fetched.
    """
    if not from_currency or not to_currency or from_currency == to_currency:
        return 1.0
    try:
        pair = yf.Ticker(f"{from_currency}{to_currency}=X")
        hist = pair.history(period="5d")
        if not hist.empty:
            return float(hist['Close'].iloc[-1])
        return None
    except Exception as e:
        print(f"Warning: Could not fetch FX rate {from_currency}->{to_currency}: {e}")
        return None

def fetch_stock_data(ticker, fallback_name=None):
    """
    Generic version of the old fetch_bti_data(): fetches price/performance
    and key valuation metrics for ANY ticker via yfinance. Handles the
    currency-mismatched EV/EBITDA case generically (returns 1.0 FX rate
    when listing and financial currency already match, so it's a no-op
    for USD-only companies like ANF/LYV/CHDN/NCLH/DIS).
    """
    data = {}
    try:
        t = yf.Ticker(ticker)

        # 1. Price and performance from history (very reliable)
        hist = t.history(period="5d")
        if not hist.empty:
            current_price = hist['Close'].iloc[-1]
            prev_price = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
            price_change = current_price - prev_price
            pct_change = (price_change / prev_price) * 100

            data['price'] = round(current_price, 2)
            data['change'] = round(price_change, 2)
            data['pct_change'] = round(pct_change, 2)
            data['volume'] = int(hist['Volume'].iloc[-1])
        else:
            data['price'] = None
            data['change'] = None
            data['pct_change'] = None
            data['volume'] = None

        # 2. Key metrics
        info = t.info
        data['name'] = info.get('longName', fallback_name or ticker)
        data['pe_ratio'] = info.get('trailingPE') or info.get('forwardPE')
        data['dividend_yield'] = info.get('dividendYield')
        data['dividend_rate'] = info.get('dividendRate')
        data['market_cap'] = info.get('marketCap')
        data['fifty_two_week_low'] = info.get('fiftyTwoWeekLow')
        data['fifty_two_week_high'] = info.get('fiftyTwoWeekHigh')

        # EV/EBITDA multiple, corrected for currency mismatches (no-op for
        # same-currency tickers since _get_fx_rate returns 1.0 in that case)
        listing_currency = info.get('currency')
        financial_currency = info.get('financialCurrency')
        market_cap = info.get('marketCap')
        total_debt = info.get('totalDebt') or 0
        total_cash = info.get('totalCash') or 0
        ebitda_raw = info.get('ebitda')

        corrected_ev_ebitda = None
        if market_cap and ebitda_raw:
            fx_rate = _get_fx_rate(financial_currency, listing_currency)
            if fx_rate:
                ebitda_converted = ebitda_raw * fx_rate
                if ebitda_converted:
                    enterprise_value = market_cap + (total_debt * fx_rate) - (total_cash * fx_rate)
                    corrected_ev_ebitda = round(enterprise_value / ebitda_converted, 2)
            else:
                print(f"Warning: FX rate unavailable for {ticker}; falling back to Yahoo's raw EV/EBITDA.")

        data['ev_ebitda_yahoo_raw'] = info.get('enterpriseToEbitda')
        data['ev_ebitda'] = corrected_ev_ebitda if corrected_ev_ebitda is not None else info.get('enterpriseToEbitda')

        if data['dividend_yield']:
            data['dividend_yield_pct'] = round(data['dividend_yield'] * 100, 2)
        else:
            data['dividend_yield_pct'] = None

        if data['market_cap']:
            data['market_cap_bn'] = round(data['market_cap'] / 1e9, 2)
        else:
            data['market_cap_bn'] = None

        # 3. yfinance stock news (fetch 5, let Gemini pick the top 2-3)
        # NOTE: current yfinance versions nest article fields under a
        # 'content' key (title/pubDate/provider/canonicalUrl); older
        # versions had them flat. We check 'content' first and fall back
        # to flat keys so this keeps working either way.
        yf_news = []
        raw_news = t.news
        if raw_news:
            for item in raw_news[:5]:
                content = item.get('content', item)
                publisher = (content.get('provider') or {}).get('displayName') or content.get('publisher')
                link = (content.get('canonicalUrl') or {}).get('url') or content.get('link')
                pub_date_raw = content.get('pubDate') or content.get('providerPublishTime')

                yf_news.append({
                    'title': content.get('title'),
                    'publisher': publisher,
                    'link': link,
                    'pubDate': _format_pub_date(pub_date_raw),
                })
        data['yfinance_news'] = yf_news

    except Exception as e:
        print(f"Error fetching yfinance data for {ticker}: {e}")
        data['error'] = str(e)

    return data

def fetch_catalyst_news(queries, limit=8, max_age_days=14, min_source_score=2):
    """
    Runs a list of Google News RSS queries (passed in per-company from
    stocks_config.py) and returns deduplicated articles that are BOTH
    recent and from a decent source, ranked best-first.

    Filtering/ranking applied before returning:
      - Date filter: articles older than `max_age_days` (or undated) are dropped.
      - Source floor: articles from sources scoring below `min_source_score`
        are dropped (content mills like insidermonkey/simplywall.st).
      - Ranking: remaining articles sorted by source quality, with recency
        as a tie-breaker, then trimmed to `limit`.

    Tunables:
      - max_age_days=14 : two-week window. Drop to 7 for a stricter brief.
      - min_source_score=2 : drops only the worst tier. Raise to 4-5 to keep
        essentially wires + major business press only.
      - limit=8 : Gemini still picks the top 2-3 for display, but a cleaner
        candidate pool means better picks.
    """
    articles = []
    seen_links = set()

    # Hard recency cutoff: anything older than this never reaches Gemini.
    now = datetime.datetime.now(datetime.timezone.utc)
    cutoff = now - datetime.timedelta(days=max_age_days)

    for query in queries:
        encoded_query = urllib.parse.quote_plus(query)
        url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
        req = urllib.request.Request(
            url,
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        try:
            with urllib.request.urlopen(req) as response:
                xml_data = response.read()
            root = ET.fromstring(xml_data)

            # Pull more per query than before (10 vs 5): filtering + ranking
            # downstream means we can afford a wider net and still end up
            # with a cleaner final set.
            for item in root.findall(".//item")[:10]:
                title = item.find("title").text if item.find("title") is not None else ""
                link = item.find("link").text if item.find("link") is not None else ""
                pub_date_str = item.find("pubDate").text if item.find("pubDate") is not None else ""

                if not link or link in seen_links:
                    continue

                pub_dt = _parse_pub_datetime(pub_date_str)

                # Date filter: drop undated or stale articles outright.
                # (Undated is dropped too — if we can't verify recency we
                # don't want a possibly-months-old article in a daily brief.)
                if pub_dt is None or pub_dt < cutoff:
                    continue

                seen_links.add(link)
                clean_title = title
                publisher = "Google News"
                if " - " in title:
                    parts = title.rsplit(" - ", 1)
                    clean_title = parts[0]
                    publisher = parts[1]

                source_score = _score_source(publisher)

                # Drop the worst-tier content mills entirely rather than
                # just downweighting them, if a floor is set.
                if source_score < min_source_score:
                    continue

                # Recency score: newer = higher, on a 0..1 scale across the window.
                age_days = (now - pub_dt).total_seconds() / 86400.0
                recency_score = max(0.0, 1.0 - (age_days / max_age_days))

                # Combined rank: source quality dominates, recency breaks ties.
                # (source_score is ~1..10; recency adds up to ~3.)
                rank = source_score + (recency_score * 3.0)

                articles.append({
                    "title": clean_title,
                    "publisher": publisher,
                    "link": link,
                    "pubDate": _format_pub_date(pub_date_str),
                    "_pub_dt": pub_dt,
                    "_source_score": source_score,
                    "_rank": rank,
                })
        except Exception as e:
            print(f"Error fetching RSS news for query '{query}': {e}")

    # Rank best-first, then trim to the limit.
    articles.sort(key=lambda a: a["_rank"], reverse=True)
    top = articles[:limit]

    # Strip internal scoring keys so the payload handed to Gemini stays clean.
    for a in top:
        a.pop("_pub_dt", None)
        a.pop("_source_score", None)
        a.pop("_rank", None)

    return top


def _rank_and_trim(articles, limit):
    """
    Shared ranking pass used by both the RSS and Finnhub fetchers so every
    feed is scored on the same scale. Expects each article dict to already
    carry a '_rank' key. Sorts best-first, trims, and strips internal keys.
    """
    articles.sort(key=lambda a: a.get("_rank", 0), reverse=True)
    top = articles[:limit]
    for a in top:
        a.pop("_pub_dt", None)
        a.pop("_source_score", None)
        a.pop("_rank", None)
    return top


def fetch_finnhub_news(ticker, limit=8, max_age_days=14, min_source_score=2):
    """
    Fetches recent company news for a ticker from Finnhub's company-news
    endpoint and returns it in the SAME shape as fetch_catalyst_news, so the
    two feeds are interchangeable and can be merged.

    Why this is a quality upgrade over the RSS path:
      - Date range is enforced SERVER-SIDE via from/to params, so stale
        articles are never even transmitted (RSS filters client-side).
      - Each article is already associated with the ticker by Finnhub, so
        we're not keyword-guessing relevance the way RSS queries do.
      - Articles come source-tagged and timestamped (unix seconds), which
        feeds cleanly into the same source-scoring we use for RSS.

    Requires the FINNHUB_API_KEY environment variable. If it's missing, or
    the request fails, this returns an empty list so the caller can fall
    back to RSS without crashing the pipeline.
    """
    api_key = os.getenv("FINNHUB_API_KEY")
    if not api_key or api_key == "YOUR_FINNHUB_API_KEY":
        print("Info: FINNHUB_API_KEY not set; skipping Finnhub news (will use RSS).")
        return []

    now = datetime.datetime.now(datetime.timezone.utc)
    frm = (now - datetime.timedelta(days=max_age_days)).strftime("%Y-%m-%d")
    to = now.strftime("%Y-%m-%d")

    url = "https://finnhub.io/api/v1/company-news"
    params = {"symbol": ticker, "from": frm, "to": to, "token": api_key}

    try:
        resp = requests.get(url, params=params, timeout=15)
        # Finnhub returns 429 on rate-limit and sometimes non-JSON bodies on
        # error, so check status before trusting the payload.
        if resp.status_code == 429:
            print(f"Warning: Finnhub rate limit hit for {ticker}; falling back to RSS.")
            return []
        resp.raise_for_status()
        raw_items = resp.json()
    except (requests.RequestException, json.JSONDecodeError, ValueError) as e:
        print(f"Warning: Finnhub news fetch failed for {ticker}: {e}")
        return []

    if not isinstance(raw_items, list):
        print(f"Warning: unexpected Finnhub response shape for {ticker}; falling back to RSS.")
        return []

    cutoff = now - datetime.timedelta(days=max_age_days)
    articles = []
    seen_links = set()

    for item in raw_items:
        # Finnhub company-news fields: headline, source, url, datetime (unix s),
        # summary, category, related.
        title = (item.get("headline") or "").strip()
        link = item.get("url") or ""
        publisher = (item.get("source") or "").strip() or "Finnhub"
        ts = item.get("datetime")

        if not title or not link or link in seen_links:
            continue

        pub_dt = _parse_pub_datetime(ts)  # handles unix int already
        # Server already bounded the range, but re-check defensively.
        if pub_dt is None or pub_dt < cutoff:
            continue

        source_score = _score_source(publisher)
        if source_score < min_source_score:
            continue

        seen_links.add(link)

        age_days = (now - pub_dt).total_seconds() / 86400.0
        recency_score = max(0.0, 1.0 - (age_days / max_age_days))
        rank = source_score + (recency_score * 3.0)

        articles.append({
            "title": title,
            "publisher": publisher,
            "link": link,
            "pubDate": _format_pub_date(ts),
            "_pub_dt": pub_dt,
            "_source_score": source_score,
            "_rank": rank,
        })

    return _rank_and_trim(articles, limit)


def fetch_company_news(ticker, catalyst_queries, limit=8,
                       max_age_days=14, min_source_score=2):
    """
    Primary news entry point. Tries Finnhub first (higher-quality, ticker-
    scoped, server-side date filtered); if it returns too few articles
    (missing key, thin ADR coverage, rate limit, etc.), tops up with the
    filtered Google News RSS feed. Results from both are deduplicated by
    link and URL-normalized title, then ranked together on the shared scale.

    This is what agent.py should call instead of fetch_catalyst_news directly.
    The old fetch_catalyst_news still exists and works standalone.

    `catalyst_queries` is the per-company RSS query list from stocks_config.py,
    used only for the fallback/top-up leg.
    """
    finnhub_items = fetch_finnhub_news(
        ticker, limit=limit, max_age_days=max_age_days,
        min_source_score=min_source_score,
    )

    # If Finnhub gave us a healthy set, we may still top up with RSS to widen
    # angle coverage, but Finnhub items keep priority. If Finnhub gave us
    # little/nothing, RSS carries the load.
    need_more = len(finnhub_items) < limit
    rss_items = []
    if need_more:
        rss_items = fetch_catalyst_news(
            catalyst_queries, limit=limit, max_age_days=max_age_days,
            min_source_score=min_source_score,
        )

    # Merge with dedup. Prefer Finnhub on collision (listed first).
    merged = []
    seen_links = set()
    seen_titles = set()

    def _title_key(t):
        return "".join(ch.lower() for ch in (t or "") if ch.isalnum())

    for item in finnhub_items + rss_items:
        link = item.get("link", "")
        tkey = _title_key(item.get("title", ""))
        if link in seen_links or (tkey and tkey in seen_titles):
            continue
        seen_links.add(link)
        if tkey:
            seen_titles.add(tkey)
        merged.append(item)

    # Finnhub items already came ranked and stripped; RSS items too. They were
    # each internally sorted, but the concatenation isn't globally sorted and
    # the internal _rank keys are gone. That's intentional: Finnhub-first
    # ordering is the priority we want. Just trim to limit.
    return merged[:limit]


def fetch_upcoming_earnings(tickers, window_days=14):
    """
    Returns a list of upcoming earnings events for the given tickers that fall
    within the next `window_days` days (default 14), sorted soonest-first.

    Data source: yfinance's Ticker.get_earnings_dates(), which returns a
    DataFrame indexed by earnings datetime. We keep only dates from today
    through today+window_days.

    IMPORTANT NUANCE: what Yahoo reports here is the EARNINGS date (the
    results release), not the 10-Q/10-K filing date. The 10-Q is typically
    filed a few days to ~2 weeks after earnings. Some dates are also
    ESTIMATED rather than company-confirmed; yfinance doesn't always expose
    a clean confirmed/estimated flag, so we conservatively label everything
    as an expected date in the report copy.

    Each returned dict:
        {
          "ticker": "DIS",
          "display_name": "The Walt Disney Company",
          "date": datetime.date(2026, 7, 22),
          "date_str": "Jul 22, 2026",
          "days_away": 6,
        }

    Returns [] if nothing falls in the window or on any fetch failure, so the
    caller can simply skip the section when the list is empty.
    """
    from stocks_config import STOCKS  # local import to avoid a cycle at module load

    now = datetime.datetime.now()
    today = now.date()
    window_end = today + datetime.timedelta(days=window_days)

    upcoming = []
    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)
            # limit keeps the call cheap; we only care about the near future.
            df = t.get_earnings_dates(limit=8)
            if df is None or df.empty:
                continue

            for idx in df.index:
                # idx is a pandas Timestamp (often tz-aware). Normalize to a date.
                try:
                    ev_date = idx.date()
                except AttributeError:
                    continue

                if today <= ev_date <= window_end:
                    display_name = STOCKS.get(ticker, {}).get("display_name", ticker)
                    upcoming.append({
                        "ticker": ticker,
                        "display_name": display_name,
                        "date": ev_date,
                        "date_str": ev_date.strftime("%b %d, %Y"),
                        "days_away": (ev_date - today).days,
                    })
        except Exception as e:
            print(f"Warning: could not fetch earnings date for {ticker}: {e}")
            continue

    # De-dupe (a ticker can occasionally return the same date twice) and sort.
    seen = set()
    deduped = []
    for e in sorted(upcoming, key=lambda x: x["date"]):
        key = (e["ticker"], e["date"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(e)

    return deduped


# FRED (Federal Reserve Economic Data) series we track for the consumer/retail
# macro section. These are the authoritative primary sources: U.S. Census
# Bureau (retail sales) and University of Michigan (sentiment), served via the
# St. Louis Fed. Each entry: series_id -> (human label, units hint, notes).
FRED_SERIES = {
    "MRTSMPCSM44000USS": ("Retail Trade sales, MoM % change", "%", ""),
    "RSXFS": ("Advance Retail Sales (ex-food services)", "$M SA", "timeliest print; advance estimate"),
    "UMCSENT": ("U. Michigan Consumer Sentiment", "index 1966Q1=100",
                "released by UMich; FRED copy delayed ~1 month at source's request"),
    "PCEC96": ("Real Personal Consumption Expenditures", "$B chained", ""),
}


def fetch_fred_indicators(series=None, per_series_points=3):
    """
    Fetches the most recent observations for a set of FRED economic series,
    for the consumer/retail macro section. Returns ONLY authoritative data
    straight from the St. Louis Fed's API (sources: U.S. Census Bureau and
    University of Michigan). Nothing here is model-generated.

    Requires FRED_API_KEY (free from https://fred.stlouisfed.org/docs/api/api_key.html).
    If the key is missing or any call fails, returns {} so the caller can build
    the macro section WITHOUT hard numbers rather than crashing or inventing them.

    Returns a dict:
        {
          "UMCSENT": {
            "label": "U. Michigan Consumer Sentiment",
            "units": "index 1966Q1=100",
            "notes": "...delayed ~1 month...",
            "latest": {"date": "2026-06-01", "value": 60.7},
            "prior": {"date": "2026-05-01", "value": 52.2},
            "change": 8.5,            # latest - prior (absolute)
            "recent": [ {date,value}, ... ]  # newest-first, per_series_points long
          },
          ...
        }
    """
    api_key = os.getenv("FRED_API_KEY")
    if not api_key or api_key == "YOUR_FRED_API_KEY":
        print("Info: FRED_API_KEY not set; macro section will omit hard economic figures.")
        return {}

    series = series or FRED_SERIES
    out = {}

    for series_id, meta in series.items():
        label, units, notes = meta
        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "sort_order": "desc",       # newest first
            "limit": max(per_series_points, 2),
        }
        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            payload = resp.json()
        except (requests.RequestException, json.JSONDecodeError, ValueError) as e:
            print(f"Warning: FRED fetch failed for {series_id}: {e}")
            continue

        obs = payload.get("observations", [])
        # FRED uses "." for missing values; filter those out.
        clean = []
        for o in obs:
            v = o.get("value")
            if v in (None, ".", ""):
                continue
            try:
                clean.append({"date": o.get("date"), "value": float(v)})
            except ValueError:
                continue

        if not clean:
            continue

        entry = {
            "label": label,
            "units": units,
            "notes": notes,
            "latest": clean[0],
            "recent": clean[:per_series_points],
        }
        if len(clean) > 1:
            entry["prior"] = clean[1]
            entry["change"] = round(clean[0]["value"] - clean[1]["value"], 2)
        out[series_id] = entry

    return out


def fetch_fred_series_history(series_id, limit=24):
    """
    Fetches up to `limit` most recent observations for a single FRED series,
    oldest-first, for charting a trend. (fetch_fred_indicators grabs just
    enough points for a latest-vs-prior comparison; this pulls a longer run
    for the macro chart.)

    Returns a list of {"date": "YYYY-MM-DD", "value": float}, oldest first,
    or [] if FRED_API_KEY is missing or the request fails.
    """
    api_key = os.getenv("FRED_API_KEY")
    if not api_key or api_key == "YOUR_FRED_API_KEY":
        return []

    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "sort_order": "desc",
        "limit": limit,
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        payload = resp.json()
    except (requests.RequestException, json.JSONDecodeError, ValueError) as e:
        print(f"Warning: FRED history fetch failed for {series_id}: {e}")
        return []

    clean = []
    for o in payload.get("observations", []):
        v = o.get("value")
        if v in (None, ".", ""):
            continue
        try:
            clean.append({"date": o.get("date"), "value": float(v)})
        except ValueError:
            continue

    clean.reverse()  # FRED returns newest-first; the chart wants oldest-first
    return clean


# The five series shown as KPI tiles on the macro chart. Each has wildly
# different units (dollars, percent, an index) and two are quarterly rather
# than monthly, so they're rendered as a row of stat tiles (one number +
# sparkline each) rather than lines sharing one axis.
#   unit          -- how to format the headline value
#   delta_unit    -- "pct" (percent change) or "pts" (raw point change) for
#                    the vs-N-readings-ago comparison
#   good_direction -- which direction of travel is good news ("up"/"down"),
#                    or None when the series doesn't support that framing
#                    (e.g. rising revolving credit can reflect either strong
#                    spending or growing household leverage)
MACRO_KPI_SERIES = {
    "RSFSXMV": {
        "label": "Retail Sales, ex. Motor Vehicles & Parts",
        "unit": "dollars_millions",
        "delta_unit": "pct",
        "good_direction": "up",
    },
    "REVOLSL": {
        "label": "Revolving Consumer Credit",
        "unit": "dollars_millions",
        "delta_unit": "pct",
        "good_direction": None,
    },
    "DRCCLACBS": {
        "label": "Credit Card Delinquency Rate",
        "unit": "percent",
        "delta_unit": "pts",
        "good_direction": "down",
    },
    "CPILFESL": {
        "label": "Core CPI, ex. Food & Energy",
        "unit": "index",
        "delta_unit": "pct",
        "good_direction": "down",
    },
    "UMCSENT": {
        "label": "U. Michigan Consumer Sentiment",
        "unit": "index",
        "delta_unit": "pts",
        "good_direction": "up",
    },
}


def _format_compact_dollars(value_millions):
    """Formats a raw FRED dollar value (already in $ millions) as $M/$B/$T."""
    abs_v = abs(value_millions)
    if abs_v >= 1_000_000:
        return f"${value_millions / 1_000_000:.2f}T"
    if abs_v >= 1_000:
        return f"${value_millions / 1_000:.1f}B"
    return f"${value_millions:.1f}M"


def _format_kpi_value(value, unit):
    """Formats a series' latest value for its stat-tile headline number."""
    if unit == "dollars_millions":
        return _format_compact_dollars(value)
    if unit == "percent":
        return f"{value:.2f}%"
    if unit == "index":
        return f"{value:,.1f}"
    return f"{value:.2f}"


def _format_kpi_delta(latest, oldest, delta_unit):
    """
    Compares the latest reading to the oldest of the fetched readings.
    Returns (display_text, direction) where direction is "up"/"down"/"flat".
    """
    diff = latest - oldest
    if delta_unit == "pct":
        if oldest == 0:
            return "n/a", "flat"
        pct = (diff / abs(oldest)) * 100
        direction = "up" if pct > 0.05 else "down" if pct < -0.05 else "flat"
        arrow = "▲" if direction == "up" else "▼" if direction == "down" else "–"
        return f"{arrow} {pct:+.1f}%", direction

    direction = "up" if diff > 0.005 else "down" if diff < -0.005 else "flat"
    arrow = "▲" if direction == "up" else "▼" if direction == "down" else "–"
    return f"{arrow} {diff:+.2f} pts", direction


def _kpi_delta_color(direction, good_direction):
    """Green when the move is good news, red when bad, muted gray otherwise."""
    if direction == "flat" or good_direction is None:
        return "#64748b"
    return "#16a34a" if direction == good_direction else "#dc2626"


def generate_macro_chart(output_path, series_ids=None, points=3):
    """
    Builds a row of consumer/retail KPI stat tiles for the macro section: one
    tile per series, each showing the latest reading, its change vs. the
    oldest of the `points` most recent available readings, and a small
    sparkline. All data comes straight from FRED (Census Bureau, Federal
    Reserve Board G.19, and BLS via the St. Louis Fed) -- nothing here is
    model-generated.

    Note on frequency: `points` counts AVAILABLE READINGS, not calendar
    months -- DRCCLACBS is a quarterly series, so its 3 "most recent readings"
    span roughly 3 quarters, not 3 months. The other four series are monthly.

    Returns True and saves a PNG on success, or False (no file written) if
    FRED_API_KEY is missing or none of the series return data.
    """
    series_ids = list(series_ids or MACRO_KPI_SERIES.keys())

    histories = {}
    for sid in series_ids:
        hist = fetch_fred_series_history(sid, limit=points)
        if hist:
            histories[sid] = hist

    if not histories:
        print("Macro chart: no FRED history available; skipping chart.")
        return False

    print("Generating consumer/retail macro KPI tiles...")
    ncols = 3
    nrows = -(-len(series_ids) // ncols)  # ceil division
    fig, axes = plt.subplots(nrows, ncols, figsize=(9, 2.9 * nrows))
    fig.set_facecolor('#f8fafc')
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    fig.suptitle("Consumer & Retail: Key Macro Indicators (FRED)",
                  fontsize=13, fontweight='bold', color='#1e293b', y=0.99)

    for i, sid in enumerate(series_ids):
        ax = axes_flat[i]
        ax.axis('off')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        meta = MACRO_KPI_SERIES.get(sid, {
            "label": sid, "unit": "index", "delta_unit": "pts", "good_direction": None,
        })

        card = FancyBboxPatch((0.02, 0.06), 0.96, 0.88,
                               boxstyle="round,pad=0,rounding_size=0.06",
                               transform=ax.transAxes, facecolor='#ffffff',
                               edgecolor='#e2e8f0', linewidth=1, zorder=0)
        ax.add_patch(card)

        hist = histories.get(sid)
        if not hist:
            ax.text(0.5, 0.5, f"{meta['label']}\n(no data)", transform=ax.transAxes,
                     ha='center', va='center', fontsize=9, color='#94a3b8')
            continue

        latest = hist[-1]["value"]

        # Wrap to a fixed width so every tile's title occupies the same
        # two-line block regardless of label length, keeping the row aligned
        # and every label safely inside its card.
        label_lines = "\n".join(textwrap.wrap(meta["label"], width=26))
        ax.text(0.08, 0.86, label_lines, transform=ax.transAxes, fontsize=8.3,
                 color='#64748b', fontweight='bold', va='top', linespacing=1.3)
        ax.text(0.08, 0.58, _format_kpi_value(latest, meta["unit"]), transform=ax.transAxes,
                 fontsize=17, color='#1A365D', fontweight='bold', va='top')

        if len(hist) >= 2:
            oldest = hist[0]["value"]
            delta_text, direction = _format_kpi_delta(latest, oldest, meta["delta_unit"])
            color = _kpi_delta_color(direction, meta.get("good_direction"))
            ax.text(0.08, 0.30, delta_text, transform=ax.transAxes, fontsize=10,
                     color=color, fontweight='bold', va='top')
            ax.text(0.08, 0.14, f"vs {hist[0]['date']}", transform=ax.transAxes,
                     fontsize=7, color='#94a3b8', va='top')

            spark_ax = ax.inset_axes([0.60, 0.16, 0.34, 0.48])
            xs = list(range(len(hist)))
            ys = [d["value"] for d in hist]
            spark_ax.plot(xs, ys, color='#cbd5e1', linewidth=2, zorder=1)
            if len(xs) > 1:
                spark_ax.plot(xs[:-1], ys[:-1], 'o', color='#cbd5e1', markersize=4, zorder=2)
            spark_ax.plot(xs[-1], ys[-1], 'o', color='#900018', markersize=6, zorder=3,
                           markeredgecolor='#ffffff', markeredgewidth=1.5)
            spread = max(ys) - min(ys)
            pad = spread * 0.3 if spread else (abs(ys[-1]) * 0.05 + 0.01)
            spark_ax.set_ylim(min(ys) - pad, max(ys) + pad)
            spark_ax.set_xlim(-0.6, len(xs) - 0.4)
            spark_ax.axis('off')
        else:
            ax.text(0.08, 0.34, "single reading available", transform=ax.transAxes,
                     fontsize=8, color='#94a3b8', va='top')

    for j in range(len(series_ids), len(axes_flat)):
        axes_flat[j].axis('off')

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    print(f"Macro chart saved successfully to {output_path}")
    return True


# Retail bellwethers and macro-consumer queries for the sector section. These
# feed the SAME filtered/scored news pipeline (fetch_company_news) so the macro
# analysis is grounded in recent, reputable, dated articles -- not model memory.
RETAIL_MACRO_QUERIES = [
    "Walmart earnings consumer outlook guidance",
    "Target Costco consumer spending results",
    "Amazon retail consumer demand outlook",
    "US consumer spending retail sales report",
    "consumer sentiment inflation spending shift",
    "private label store brand trade down consumer",
    "retailer pricing strategy discount promotion margins",
]


def fetch_retail_macro_news(limit=14, max_age_days=14, min_source_score=3):
    """
    Gathers recent, source-scored news for the consumer/retail macro section
    using the existing news pipeline. Uses a HIGHER source floor (min_source_score=3)
    than the per-stock feed, because macro claims should lean on quality outlets.

    Reuses fetch_catalyst_news (the filtered/ranked RSS path) across the
    RETAIL_MACRO_QUERIES set. Finnhub is company-scoped, so RSS is the right
    tool for these thematic/macro queries.
    """
    return fetch_catalyst_news(
        RETAIL_MACRO_QUERIES,
        limit=limit,
        max_age_days=max_age_days,
        min_source_score=min_source_score,
    )


def generate_comparison_chart(output_path, tickers):
    """
    Generates a 5-day normalized (% change) performance chart for an
    arbitrary group of tickers.

    `tickers` is a dict of {display_label: ticker_symbol}, so it now scales
    to any number of stocks instead of the old fixed BTI/MO/PM trio.
    """
    print("Generating 5-day stock performance chart...")

    # Fetch each ticker's close prices first, then keep only the days where
    # *every* ticker has a price so lines stay aligned on a shared date axis.
    closes_by_label = {}
    for label, ticker_sym in tickers.items():
        t = yf.Ticker(ticker_sym)
        hist = t.history(period="7d")
        hist = hist.dropna(subset=['Close'])
        if hist.empty:
            print(f"Warning: No historical data found for {ticker_sym}")
            continue
        series = hist['Close'].copy()
        series.index = series.index.normalize()
        closes_by_label[label] = series

    if not closes_by_label:
        print("Error: No historical data available for any ticker; skipping chart.")
        return

    combined = pd.DataFrame(closes_by_label).dropna(how='any')
    combined = combined.tail(5)

    if combined.empty:
        print("Error: No overlapping trading days found across tickers; skipping chart.")
        return

    plt.figure(figsize=(9, 5))
    plt.gcf().set_facecolor('#f8fafc')
    ax = plt.axes()
    ax.set_facecolor('#ffffff')

    # Auto-assign distinct colors so this scales to any number of tickers
    # (the old version hardcoded a color per fixed company name).
    cmap = plt.get_cmap('tab10')
    colors = {label: cmap(i % 10) for i, label in enumerate(combined.columns)}

    dates = [d.strftime('%m/%d') for d in combined.index]

    for label in combined.columns:
        closes = combined[label]
        base_price = closes.iloc[0]
        pct_change = ((closes / base_price) - 1) * 100

        plt.plot(dates, pct_change.values,
                 label=label,
                 color=colors[label],
                 linewidth=2,
                 marker='o')

    plt.title("Portfolio Group: Last 5 Trading Days Performance",
              fontsize=12, fontweight='bold', color='#1e293b', pad=10)
    plt.xlabel("Date", fontsize=10, color='#64748b')
    plt.ylabel("Performance Change (%)", fontsize=10, color='#64748b')

    plt.grid(True, linestyle=':', alpha=0.6, color='#cbd5e1')
    plt.axhline(0, color='#94a3b8', linewidth=0.8, linestyle='-')

    plt.legend(frameon=True, facecolor='#ffffff', edgecolor='#e2e8f0', loc='best', fontsize=8, ncol=2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cbd5e1')
    ax.spines['bottom'].set_color('#cbd5e1')
    ax.tick_params(axis='both', colors='#64748b')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor=plt.gcf().get_facecolor(), edgecolor='none')
    plt.close()
    print(f"Chart saved successfully to {output_path}")

if __name__ == "__main__":
    from stocks_config import STOCKS
    print("Fetching stock metrics for all tickers...")
    for ticker, info in STOCKS.items():
        data = fetch_stock_data(ticker, info["display_name"])
        print(f"{ticker} EV/EBITDA: {data.get('ev_ebitda')}")

    chart_file = os.path.join(os.path.dirname(__file__), "test_chart.png")
    generate_comparison_chart(chart_file, {t: t for t in STOCKS})
