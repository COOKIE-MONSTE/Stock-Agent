import yfinance as yf
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import datetime
import os
import pandas as pd
import matplotlib
# Use the non-interactive Agg backend to avoid GUI window popup issues on Windows/servers
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
        yf_news = []
        raw_news = t.news
        if raw_news:
            for item in raw_news[:5]:
                yf_news.append({
                    'title': item.get('title'),
                    'publisher': item.get('publisher'),
                    'link': item.get('link'),
                    'type': item.get('type')
                })
        data['yfinance_news'] = yf_news

    except Exception as e:
        print(f"Error fetching yfinance data for {ticker}: {e}")
        data['error'] = str(e)

    return data

def fetch_catalyst_news(queries, limit=12):
    """
    Generic version of the old fetch_regulatory_news(): runs a list of
    Google News RSS queries (passed in per-company from stocks_config.py)
    and returns deduplicated articles.
    """
    articles = []
    seen_links = set()

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

            for item in root.findall(".//item")[:5]:  # Limit to 5 per query
                title = item.find("title").text if item.find("title") is not None else ""
                link = item.find("link").text if item.find("link") is not None else ""
                pub_date_str = item.find("pubDate").text if item.find("pubDate") is not None else ""

                if link not in seen_links:
                    seen_links.add(link)
                    clean_title = title
                    publisher = "Google News"
                    if " - " in title:
                        parts = title.rsplit(" - ", 1)
                        clean_title = parts[0]
                        publisher = parts[1]

                    articles.append({
                        "title": clean_title,
                        "publisher": publisher,
                        "link": link,
                        "pubDate": pub_date_str
                    })
        except Exception as e:
            print(f"Error fetching RSS news for query '{query}': {e}")

    return articles[:limit]

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
