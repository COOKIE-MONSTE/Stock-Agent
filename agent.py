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
You are an elite equity analyst producing one section of a daily multi-stock
investor briefing email. You will be given data for ONE company at a time.

Your output MUST be a single self-contained HTML fragment (a <div>...</div>)
with inline CSS styles only. Do NOT include <html>, <head>, or <body> tags,
and do not output markdown backticks or any text outside the <div>.

Visual style (match this exactly so every company's card looks consistent):
- White card, 16px border-radius, 1px solid #e2e8f0 border, 20px padding, 18px bottom margin, font-family 'Segoe UI', Arial, sans-serif.
- Header row inside the card: company name + ticker in bold #1A365D, with current price, daily % change (use #16a34a for positive, #dc2626 for negative), and EV/EBITDA multiple shown on the same line or right-aligned.

Structure inside the card:
1. Executive Summary: exactly 2 sentences summarizing the stock's current state.
2. Key Catalysts & Risks: EXACTLY three bullet points. These should be the three most relevant, market-moving items for THIS SPECIFIC company drawn from the news provided -- they may be regulatory, legal/antitrust, demand or guidance related, competitive, or M&A. Do not force categories that don't fit; choose whichever three are genuinely most relevant based on the supplied data.
3. Key News: the top 2-3 most critical, highest-impact headlines (prefer trustworthy sources like Reuters, Bloomberg, WSJ) as active clickable hyperlinks. Show each headline's publish date next to it (small, muted text, e.g. "Jun 16, 2026") using the date supplied in the data -- never invent a date.

Be factual and concise. Base everything on the data given; never fabricate figures, news, or sources.
"""

def generate_stock_section(ticker, info, stock_data, news_items):
    """
    Calls Gemini once for a single stock and returns just its HTML card
    (a fragment, not a full document) at the same analytical depth the
    original single-stock report had.
    """
    client = get_gemini_client()
    display_name = info.get("display_name", ticker)

    stock_info_str = f"""
    Ticker: {ticker} ({display_name})
    Current Price: ${stock_data.get('price', 'N/A')}
    Daily Change: ${stock_data.get('change', 'N/A')} ({stock_data.get('pct_change', 'N/A')}%)
    EV/EBITDA Multiple: {stock_data.get('ev_ebitda', 'N/A')}
    Dividend Yield: {stock_data.get('dividend_yield_pct', 'N/A')}%
    Market Cap: {stock_data.get('market_cap_bn', 'N/A')} Billion USD
    """

    yfinance_news_str = "\n".join([
        f"- **{n['title']}** (Source: {n['publisher']}, Date: {n['pubDate']}) - Link: {n['link']}"
        for n in stock_data.get('yfinance_news', [])
    ]) or "(no general market news returned)"

    catalyst_news_str = "\n".join([
        f"- **{n['title']}** (Source: {n['publisher']}) - Link: {n['link']} - Date: {n['pubDate']}"
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

    Produce the HTML card for this company following the structure and
    styling rules exactly.
    """

    max_retries = 3
    retry_delay = 3
    response = None

    for attempt in range(max_retries):
        try:
            print(f"Requesting AI analysis for {ticker} (Attempt {attempt + 1}/{max_retries})...")
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=STOCK_SECTION_SYSTEM_INSTRUCTION,
                    temperature=0.1,
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
            {body}
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
