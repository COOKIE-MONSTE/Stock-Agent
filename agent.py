import os
import time
import datetime
from google import genai
from google.genai import types
from google.genai.errors import ServerError, APIError
from dotenv import load_dotenv
import data_fetcher
from stocks_config import STOCKS

# Load configuration
load_dotenv()

# Model selection. gemini-3.5-flash is the current-generation Flash model
# (free tier, far higher daily quota than the old 2.5-flash 20/day limit) and
# is well-suited to this synthesis/analysis task. Override via env if desired.
# If Google ever retires this string, swap it here in one place.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")

def get_gemini_client():
    """
    Initializes and returns the Google GenAI Client.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "YOUR_GEMINI_API_KEY":
        raise ValueError("Missing GEMINI_API_KEY in environment or .env file.")
    return genai.Client()

def clean_html_output(text):
    """
    Cleans any markdown wrapper blocks (e.g., ```html ... ```) from Gemini's response.
    """
    text = text.strip()
    if text.startswith("```html"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()

# This is now company-agnostic: Gemini is asked to find the three most
# relevant catalysts/risks FOR WHATEVER COMPANY it's given, rather than
# being told fixed categories (the old prompt's "FDA / illicit vape / ex-US"
# only made sense for BTI). Per-company search queries in stocks_config.py
# still steer the *kind* of news each company gets, so quality/relevance is
# preserved even though the prompt itself is generic.
STOCK_SECTION_SYSTEM_INSTRUCTION = """
You are a senior equity analyst writing one company's section of a daily
investor briefing. Your job is NOT to summarize headlines -- it is to explain
WHY the stock is doing what it's doing today, connecting price action to
concrete, verifiable drivers.

=== YOUR CORE TASK ===
For the company you're given, the single most important thing you produce is a
causal explanation of today's price move. Ask yourself, in order:
1. Is today's move explained by a COMPANY-SPECIFIC event in the data
   (earnings, guidance, an analyst action, a regulatory/legal development,
   M&A, a product or operational event)?
2. If not, is it explained by a MACRO or SECTOR force (rates, oil, consumer-
   spending data, a sector-wide move, geopolitics) that would push peers the
   same way?
3. If NEITHER the company news nor a plausible macro/sector driver in the data
   explains the move, SAY SO PLAINLY -- e.g. "Today's 2% gain has no clear
   catalyst in the available data and appears to be noise or drift." Do not
   invent a narrative to fill the gap. Admitting "no clear catalyst" is a
   correct, valuable answer and is strongly preferred over manufactured
   causation.

Be specific and evidential. Tie claims to the dated items you were given.
"The stock rose 4.4%, its context supported by X (Reuters, Jul 14)" is good;
"the stock rose on positive sentiment" is a failure. Never fabricate figures,
events, dates, or sources; use only what's in the supplied data.

Distinguish clearly between what you KNOW (it's in the data) and what you're
INFERRING (a plausible link). Hedge inferences honestly ("this likely
reflects...", "consistent with...") rather than asserting them as fact.

=== OUTPUT FORMAT ===
Output a single self-contained HTML fragment (one <div>...</div>) with inline
CSS only. No <html>/<head>/<body> tags, no markdown backticks, no text outside
the div. Use <strong> for emphasis, never markdown asterisks.

Card shell: white background, 16px border-radius, 1px solid #e2e8f0 border,
20px padding, 18px bottom margin, font-family 'Segoe UI', Arial, sans-serif.
Header row: company name + ticker in bold #1A365D; current price and daily %
change on the same row (green #16a34a if positive, red #dc2626 if negative);
EV/EBITDA multiple. Keep the header compact and consistent.

Inside the card, in this order:
1. "What moved the stock" -- 2-4 sentences of causal analysis per the CORE TASK
   above. This is the heart of the card; give it the most depth. It is fine for
   this section to be longer for a stock with a real catalyst and shorter for
   one that's just drifting.
2. "Drivers & risks" -- the genuinely material catalysts and risks, as a short
   bullet list. Include ONLY items that clear a materiality bar (could plausibly
   move the stock). Prefer two strong points over three padded ones; you may
   give anywhere from one to four. Do NOT pad to hit a number. Where useful,
   label each as a near-term catalyst vs. a standing risk.
3. "Key news" -- the 2-3 highest-impact headlines as clickable <a> links,
   preferring reputable sources, each with its supplied publish date in small
   muted text. Never invent a date; use only dates provided.

Depth and honesty matter more than uniformity across cards. A shorter, sharper,
correctly-hedged card beats a longer padded one.
"""

def generate_stock_section(ticker, info, stock_data, news_items):
    """
    Calls Gemini once for a single stock and returns just its HTML card
    (a fragment, not a full document) at the same analytical depth the
    original single-stock report had.
    """
    client = get_gemini_client()
    display_name = info.get("display_name", ticker)

    # Frame the move explicitly so the model reasons about direction/magnitude
    # rather than restating the number. Compute a plain-language descriptor.
    pct = stock_data.get('pct_change')
    if isinstance(pct, (int, float)):
        direction = "up" if pct > 0 else "down" if pct < 0 else "flat"
        magnitude = ("a large" if abs(pct) >= 4 else
                     "a moderate" if abs(pct) >= 1.5 else
                     "a small" if abs(pct) > 0 else "essentially no")
        move_str = f"The stock is {direction} {pct}% today ({magnitude} move)."
    else:
        move_str = "Today's move is unavailable."

    stock_info_str = f"""
    Ticker: {ticker} ({display_name})
    Current Price: ${stock_data.get('price', 'N/A')}
    Daily Change: ${stock_data.get('change', 'N/A')} ({stock_data.get('pct_change', 'N/A')}%)
    {move_str}
    52-Week Range: ${stock_data.get('fifty_two_week_low', 'N/A')} - ${stock_data.get('fifty_two_week_high', 'N/A')}
    Volume (today): {stock_data.get('volume', 'N/A')}
    EV/EBITDA Multiple: {stock_data.get('ev_ebitda', 'N/A')}
    Dividend Yield: {stock_data.get('dividend_yield_pct', 'N/A')}%
    Market Cap: {stock_data.get('market_cap_bn', 'N/A')} Billion USD
    """

    yfinance_news_str = "\n".join([
        f"- {n['title']} (Source: {n['publisher']}, Date: {n['pubDate']}) - Link: {n['link']}"
        for n in stock_data.get('yfinance_news', [])
    ]) or "(no general market news returned)"

    catalyst_news_str = "\n".join([
        f"- {n['title']} (Source: {n['publisher']}, Date: {n['pubDate']}) - Link: {n['link']}"
        for n in news_items
    ]) or "(no catalyst news returned)"

    prompt = f"""
    Company data for {display_name} ({ticker}):

    === Stock Metrics ===
    {stock_info_str}

    === General Market News ===
    {yfinance_news_str}

    === Catalyst / Regulatory / Sector News ===
    {catalyst_news_str}

    Write this company's briefing card. Lead with your causal explanation of
    today's move: identify the most likely driver (company-specific first, then
    macro/sector), tie it to specific dated items above, and clearly flag when a
    link is an inference rather than a stated fact. If nothing in the data
    explains the move, say so directly rather than inventing a reason. Then give
    the material drivers/risks and the key news links, following the format
    rules. Prioritize analytical depth and honesty over matching other cards.
    """

    max_retries = 3
    retry_delay = 3
    response = None

    for attempt in range(max_retries):
        try:
            print(f"Requesting AI analysis for {ticker} (Attempt {attempt + 1}/{max_retries})...")
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=STOCK_SECTION_SYSTEM_INSTRUCTION,
                    temperature=0.5,
                )
            )
            break
        except (ServerError, APIError) as e:
            if attempt == max_retries - 1:
                print(f"Failed to contact Gemini for {ticker} after multiple retries.")
                raise e
            print(f"Gemini API experiencing high demand (503) for {ticker}. Retrying in {retry_delay}s...")
            time.sleep(retry_delay)
            retry_delay *= 2
        except Exception as e:
            raise e

    return clean_html_output(response.text)

def build_earnings_section(events, window_days=14):
    """
    Renders the 'Upcoming Earnings' HTML block from a list of events
    (as returned by data_fetcher.fetch_upcoming_earnings).

    Returns an EMPTY STRING when there are no events, so the caller can
    concatenate unconditionally and the section simply vanishes from the
    email when nothing is scheduled in the window. This is deterministic
    Python-built HTML -- no Gemini call -- so it costs no quota and can't
    hallucinate a date.
    """
    if not events:
        return ""

    rows = []
    for e in events:
        days = e["days_away"]
        when = "today" if days == 0 else ("tomorrow" if days == 1 else f"in {days} days")
        rows.append(f"""
        <tr>
            <td style="padding:8px 12px;font-weight:bold;color:#1A365D;">{e['ticker']}</td>
            <td style="padding:8px 12px;color:#334155;">{e['display_name']}</td>
            <td style="padding:8px 12px;color:#334155;white-space:nowrap;">{e['date_str']}</td>
            <td style="padding:8px 12px;color:#64748b;white-space:nowrap;">{when}</td>
        </tr>""")

    rows_html = "".join(rows)
    plural = "report" if len(events) == 1 else "reports"

    return f"""
    <div style="background-color:#ffffff;border:1px solid #e2e8f0;border-radius:16px;padding:20px;margin-bottom:18px;font-family:'Segoe UI', Arial, sans-serif;">
        <h2 style="margin:0 0 4px;font-size:16px;color:#1A365D;">📅 Upcoming Earnings (next {window_days} days)</h2>
        <p style="margin:0 0 12px;font-size:12px;color:#64748b;">
            {len(events)} portfolio {plural} scheduled. Dates are expected earnings-release dates;
            the related 10-Q/10-K is typically filed shortly after and some dates may still be estimates.
        </p>
        <table style="width:100%;border-collapse:collapse;font-size:13px;">
            <thead>
                <tr style="text-align:left;border-bottom:1px solid #e2e8f0;color:#94a3b8;font-size:11px;text-transform:uppercase;">
                    <th style="padding:6px 12px;">Ticker</th>
                    <th style="padding:6px 12px;">Company</th>
                    <th style="padding:6px 12px;">Expected Date</th>
                    <th style="padding:6px 12px;">When</th>
                </tr>
            </thead>
            <tbody>{rows_html}</tbody>
        </table>
    </div>
    """


MACRO_SECTION_SYSTEM_INSTRUCTION = """
You are an expert equity research analyst and retail strategist writing the
macro/sector section of a daily investor briefing focused on consumer & retail.

=== NON-NEGOTIABLE GROUNDING RULES ===
You will be given two evidence sources: (1) recent, dated news articles, and
(2) official economic figures from FRED (U.S. Census Bureau retail sales and
University of Michigan sentiment). These are your ONLY permitted sources of fact.

- Every specific statistic (a sentiment index level, a retail-sales % change,
  a dollar figure) MUST come from the FRED data block provided. NEVER state a
  numeric economic figure that is not in that block. If you don't have a number,
  describe the direction/dynamic qualitatively instead.
- Every claim about a company or a recent event MUST trace to a provided news
  item. Do not reference earnings calls, executive quotes, or corporate moves
  that aren't in the supplied news.
- When the FRED note says a series is delayed (e.g. Michigan sentiment lags ~1
  month), acknowledge that lag rather than implying the figure is today's.
- If the evidence is thin on some sub-topic, say so briefly rather than filling
  the gap with plausible-sounding invention. "The provided data doesn't cover X"
  is an acceptable and preferred statement.
- Clearly separate FACT (from the data) from INTERPRETATION (your analysis).
  Hedge inferences ("this likely reflects...", "consistent with...").

=== ANALYTICAL MANDATE ===
Within those guardrails, be deep and specific. Avoid generic statements like
"consumers are feeling inflation." Instead surface concrete behavioral dynamics,
e.g. "higher-income shoppers trading down to private label to protect
discretionary budgets," WHEN the evidence supports it. Cover, without limiting
yourself to:
- Retail financial intelligence: what major retailers' recent commentary (from
  the provided news) implies about the consumer.
- Data indicators: interpret the FRED figures provided (retail sales momentum,
  sentiment direction) - what do they signal?
- K-shaped bifurcation: how low- / middle- / high-income cohorts are behaving
  differently, grounded in the evidence.
- Competitive corporate strategies: pricing/value tactics, private-label vs.
  premiumization, operational/tech levers - as reflected in the provided news.
You may add other genuinely relevant macro trends, regulatory shifts, or
emerging strategies IF supported by the evidence provided.

=== OUTPUT FORMAT ===
A single self-contained HTML fragment: one <div>...</div>, inline CSS only, no
<html>/<head>/<body>, no markdown backticks, no text outside the div. Use
<strong> for emphasis, never markdown asterisks.

Card shell: white background, 16px border-radius, 1px solid #e2e8f0 border,
20px padding, 18px bottom margin, font-family 'Segoe UI', Arial, sans-serif.
Open with a header: <h2 style="margin:0 0 12px;font-size:17px;color:#1A365D;">
Consumer & Retail: Macro & Sector Read</h2>

Use clear <h3> subheadings (color #334155, font-size 14px), concise paragraphs,
and <ul> bullets for specifics. Where you cite a FRED figure, present it as a
small data callout (e.g. a <div> with a light #f1f5f9 background). Keep it
skimmable and professional. Depth and evidential grounding beat length.
"""


def _format_fred_for_prompt(fred_data):
    """Turn the FRED dict into a compact text block for the prompt."""
    if not fred_data:
        return "(No official economic figures available this run — do NOT state any specific economic statistics; discuss dynamics qualitatively only.)"
    lines = []
    for sid, d in fred_data.items():
        latest = d.get("latest", {})
        prior = d.get("prior", {})
        chg = d.get("change")
        piece = f"- {d['label']} [{d['units']}]: latest {latest.get('value')} ({latest.get('date')})"
        if prior:
            piece += f"; prior {prior.get('value')} ({prior.get('date')})"
        if chg is not None:
            piece += f"; change {chg:+}"
        if d.get("notes"):
            piece += f". Note: {d['notes']}"
        lines.append(piece)
    return "\n".join(lines)


def generate_macro_section(macro_news, fred_data):
    """
    Produces the consumer/retail macro section via one grounded Gemini call.
    Returns the HTML fragment, or "" on failure so the report can omit it.

    Grounding: FRED figures are the ONLY permitted numeric source; news items
    are the ONLY permitted event source. The system instruction enforces this.
    """
    if not macro_news and not fred_data:
        print("Macro section: no news and no FRED data available; skipping section.")
        return ""

    news_str = "\n".join([
        f"- {n['title']} (Source: {n['publisher']}, Date: {n['pubDate']}) - Link: {n['link']}"
        for n in macro_news
    ]) or "(no macro/retail news returned this run)"

    fred_str = _format_fred_for_prompt(fred_data)

    prompt = f"""
    Build the Consumer & Retail macro/sector section using ONLY the evidence below.

    === OFFICIAL ECONOMIC FIGURES (FRED — U.S. Census Bureau & U. Michigan) ===
    These are the ONLY numbers you may cite. Do not state any economic statistic
    not listed here.
    {fred_str}

    === RECENT CONSUMER / RETAIL NEWS (dated, source-scored) ===
    Every company/event claim must trace to one of these items.
    {news_str}

    Write the section now, following the grounding rules and format exactly.
    Prioritize specific, evidenced behavioral insight over generic commentary,
    and explicitly note where the evidence is thin rather than inventing detail.
    """

    max_retries = 3
    retry_delay = 3
    response = None
    for attempt in range(max_retries):
        try:
            print(f"Requesting macro/sector analysis (Attempt {attempt + 1}/{max_retries})...")
            response = get_gemini_client().models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=MACRO_SECTION_SYSTEM_INSTRUCTION,
                    temperature=0.5,
                )
            )
            break
        except (ServerError, APIError) as e:
            if attempt == max_retries - 1:
                print(f"Macro section: Gemini unavailable after retries: {e}")
                return ""
            print(f"Gemini high demand for macro section. Retrying in {retry_delay}s...")
            time.sleep(retry_delay)
            retry_delay *= 2
        except Exception as e:
            print(f"Macro section failed: {e}")
            return ""

    return clean_html_output(response.text)


def generate_portfolio_report(tickers=None):
    """
    Builds the full multi-stock HTML email:
      1. Fetches data + tailored catalyst news for each ticker.
      2. Calls Gemini once per ticker for its analysis card (same depth as
         the original single-stock report).
      3. Wraps every card in one consistent, Python-built HTML shell
         (shared header/chart/footer) so the email reads as one report
         instead of stitched-together documents.

    A failure on one ticker (data fetch or Gemini call) no longer kills the
    whole report -- that ticker gets a placeholder card and the rest of the
    pipeline continues.

    Returns: (full_html, summary) where summary is a list of
    {"ticker", "price", "pct_change"} dicts, used for the email subject line.
    """
    if tickers is None:
        tickers = list(STOCKS.keys())

    current_date = datetime.datetime.now().strftime("%B %d, %Y")
    sections_html = []
    summary = []

    for ticker in tickers:
        info = STOCKS.get(ticker, {
            "display_name": ticker,
            "catalyst_queries": [f"{ticker} stock news regulation"],
        })
        try:
            print(f"Gathering data for {ticker}...")
            stock_data = data_fetcher.fetch_stock_data(ticker, fallback_name=info["display_name"])
            if 'error' in stock_data:
                print(f"Warning: {ticker} stock data fetch had issues: {stock_data['error']}")

            news_items = data_fetcher.fetch_company_news(
                ticker, info["catalyst_queries"]
            )

            section_html = generate_stock_section(ticker, info, stock_data, news_items)
            sections_html.append(section_html)
            summary.append({
                "ticker": ticker,
                "price": stock_data.get("price"),
                "pct_change": stock_data.get("pct_change"),
            })
        except Exception as e:
            print(f"Warning: failed to build section for {ticker}, inserting placeholder: {e}")
            sections_html.append(
                f'<div style="background-color:#ffffff;border:1px solid #e2e8f0;'
                f'border-radius:16px;padding:20px;margin-bottom:18px;color:#1e293b;'
                f'font-family:\'Segoe UI\', Arial, sans-serif;">'
                f'<strong>{ticker}</strong>: data temporarily unavailable for today\'s report.</div>'
            )
            summary.append({"ticker": ticker, "price": None, "pct_change": None})

    header_html = f"""
    <div style="background-color:#1A365D;color:#ffffff;padding:24px;border-radius:16px 16px 0 0;font-family:'Segoe UI', Arial, sans-serif;">
        <h1 style="margin:0;font-size:20px;">Daily Portfolio Update</h1>
        <p style="margin:6px 0 0;font-size:13px;color:#cbd5e1;">{current_date}</p>
    </div>
    """

    chart_html = """
    <div style="background-color:#ffffff;padding:16px;text-align:center;">
        <img src="cid:comparison_chart" style="max-width:100%;border-radius:12px;border:1px solid #e2e8f0;" />
    </div>
    """

    # Upcoming-earnings section: only rendered if something falls in the
    # next two weeks. build_earnings_section returns "" otherwise, so the
    # block simply disappears from the email when nothing is scheduled.
    EARNINGS_WINDOW_DAYS = 14
    try:
        earnings_events = data_fetcher.fetch_upcoming_earnings(
            tickers, window_days=EARNINGS_WINDOW_DAYS
        )
        earnings_html = build_earnings_section(earnings_events, EARNINGS_WINDOW_DAYS)
        if earnings_events:
            print(f"Upcoming earnings in next {EARNINGS_WINDOW_DAYS} days: "
                  + ", ".join(f"{e['ticker']} ({e['date_str']})" for e in earnings_events))
        else:
            print(f"No portfolio earnings within {EARNINGS_WINDOW_DAYS} days; skipping section.")
    except Exception as e:
        print(f"Warning: earnings section failed, omitting it: {e}")
        earnings_html = ""

    # Consumer & retail macro/sector section. Grounded in FRED official figures
    # + recent source-scored retail news. Omitted entirely on failure.
    try:
        print("Gathering macro/retail evidence (FRED + news)...")
        fred_data = data_fetcher.fetch_fred_indicators()
        macro_news = data_fetcher.fetch_retail_macro_news()
        macro_html = generate_macro_section(macro_news, fred_data)
    except Exception as e:
        print(f"Warning: macro section failed, omitting it: {e}")
        macro_html = ""

    footer_html = """
    <div style="background-color:#f8fafc;color:#64748b;font-size:11px;padding:16px;border-radius:0 0 16px 16px;text-align:center;font-family:'Segoe UI', Arial, sans-serif;">
        This report is for informational purposes only and does not constitute investment advice.
    </div>
    """

    body = "\n".join(sections_html)
    full_html = f"""
    <html>
    <body style="background-color:#f1f5f9;font-family:'Segoe UI', Arial, sans-serif;margin:0;padding:24px;">
        <div style="max-width:680px;margin:0 auto;">
            {header_html}
            {chart_html}
            {earnings_html}
            {body}
            {macro_html}
            {footer_html}
        </div>
    </body>
    </html>
    """

    return full_html, summary

if __name__ == "__main__":
    # Test stub
    report_html, summary = generate_portfolio_report()
    print("Report generated successfully.")
    output_path = os.path.join(os.path.dirname(__file__), "last_report.html")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_html)
    print(f"Saved local copy of report to: {output_path}")
