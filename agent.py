import os
from google import genai
from google.genai import types
from dotenv import load_dotenv
import data_fetcher
import datetime

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

def generate_bti_report():
    """
    Main agent pipeline:
    1. Fetches BTI stock metrics and news.
    2. Runs analysis with Gemini.
    3. Outputs clean HTML showing EV/EBITDA, 3 custom regulatory points,
       and top news headlines alongside the comparison chart placeholder.
    """
    print("Gathering real-time market data...")
    stock_data = data_fetcher.fetch_bti_data()
    
    print("Gathering news and global regulatory updates...")
    reg_news = data_fetcher.fetch_regulatory_news()
    
    if 'error' in stock_data:
        print(f"Warning: Stock data fetch had issues: {stock_data['error']}")

    client = get_gemini_client()
    current_date = datetime.datetime.now().strftime("%B %d, %Y")
    
    # Format inputs for the prompt
    stock_info_str = f"""
    Ticker: BTI (British American Tobacco p.l.c. ADR)
    Current Price: ${stock_data.get('price', 'N/A')}
    Daily Change: ${stock_data.get('change', 'N/A')} ({stock_data.get('pct_change', 'N/A')}%)
    EV/EBITDA Multiple: {stock_data.get('ev_ebitda', 'N/A')}
    Dividend Yield: {stock_data.get('dividend_yield_pct', 'N/A')}%
    Market Cap: {stock_data.get('market_cap_bn', 'N/A')} Billion USD
    """

    yfinance_news_str = "\n".join([
        f"- **{n['title']}** (Source: {n['publisher']}) - Link: {n['link']}"
        for n in stock_data.get('yfinance_news', [])
    ])

    reg_news_str = "\n".join([
        f"- **{n['title']}** (Source: {n['publisher']}) - Link: {n['link']} - Date: {n['pubDate']}"
        for n in reg_news
    ])

    system_instruction = """
    You are an elite financial analyst and a specialized regulatory intelligence agent for the tobacco and nicotine industry.
    Your task is to analyze the provided data on British American Tobacco (BTI) and compile a daily email briefing for an investor.
    
    Your output MUST be a complete, well-formed HTML document containing inline CSS styles. 
    Do NOT output any markdown backticks (like ```html), and do not output any introductory or concluding text outside the HTML.
    
    Design constraints for the HTML email:
    - Use clean, premium modern typography (system sans-serif fonts, e.g., 'Segoe UI', Arial, sans-serif).
    - Color palette: Deep navy background for header (#1A365D), text colors (#1e293b), soft grey backgrounds (#f8fafc), and white content blocks.
    - Structure:
        1. A header displaying the report date, stock name, current price, daily percent change, and EV/EBITDA multiple.
        2. Financial Performance Chart: Include an image referencing `cid:comparison_chart` which displays the 5-day performance of BTI vs Altria & Philip Morris. Give it a nice rounded border, a light grey border, and center it.
        3. Executive Summary: 2 sentences summarizing the current state.
        4. Regulatory Analysis: Provide EXACTLY three bullet points under this section. Do NOT write more or fewer.
            - Bullet 1: Recent FDA regulation changes impacting BTI/nicotine products.
            - Bullet 2: How the illicit e-cigarette market in the US is responding to regulatory crackdowns.
            - Bullet 3: Recent regulatory changes/updates impacting BTI *outside* the US (global markets like EU, UK, or Asia).
        5. Key Market News Feed: Display only the top 2-3 most critical, high-impact news headlines (e.g. from highly trustworthy sources like Reuters, Bloomberg, WSJ, etc.) that are actually causing or likely to cause stock movement. Format them as active clickable hyperlinks.
        6. A disclaimer footer.
    """

    prompt = f"""
    Here is the daily stock and news feed data for BTI on {current_date}:
    
    === Stock Metrics ===
    {stock_info_str}
    
    === General Market News ===
    {yfinance_news_str}
    
    === Global Regulatory & Nicotine News (Google News RSS) ===
    {reg_news_str}
    
    Analyze the data and create the HTML email template. Make sure:
    - The BTI EV/EBITDA multiple is displayed prominently.
    - The comparison chart is embedded using `<img src="cid:comparison_chart" ... />` inside the HTML layout.
    - The regulatory section has EXACTLY three bullet points matching the target topics (FDA changes, US illicit vape crackdown response, and global/ex-US updates).
    - The news feed is short, containing only the top 2-3 most trustworthy and high-impact headlines with their active links.
    """

    import time
    from google.genai.errors import ServerError, APIError

    max_retries = 3
    retry_delay = 3
    response = None
    
    for attempt in range(max_retries):
        try:
            print(f"Requesting AI analysis from Gemini (Attempt {attempt + 1}/{max_retries})...")
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.1, # Keep it highly factual
                )
            )
            break
        except (ServerError, APIError) as e:
            if attempt == max_retries - 1:
                print("Failed to contact Gemini after multiple retries due to high server demand.")
                raise e
            print(f"Gemini API experiencing high demand (503). Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
            retry_delay *= 2
        except Exception as e:
            # Raise other unexpected errors immediately
            raise e
    
    html_content = clean_html_output(response.text)
    return html_content, stock_data.get('price'), stock_data.get('pct_change')

if __name__ == "__main__":
    # Test stub
    report_html, price, change = generate_bti_report()
    print("Report generated successfully.")
    output_path = os.path.join(os.path.dirname(__file__), "last_report.html")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_html)
    print(f"Saved local copy of report to: {output_path}")
