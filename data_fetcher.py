import yfinance as yf
import urllib.request
import xml.etree.ElementTree as ET
import datetime
import os
import matplotlib
# Use the non-interactive Agg backend to avoid GUI window popup issues on Windows/servers
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def fetch_bti_data():
    """
    Fetches BTI stock information, historical prices, and metrics from yfinance.
    """
    data = {}
    try:
        ticker = yf.Ticker("BTI")
        
        # 1. Fetch current price and performance from history (very reliable)
        hist = ticker.history(period="5d")
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

        # 2. Fetch key metrics
        info = ticker.info
        data['name'] = info.get('longName', 'British American Tobacco p.l.c.')
        data['pe_ratio'] = info.get('trailingPE') or info.get('forwardPE')
        data['dividend_yield'] = info.get('dividendYield') 
        data['dividend_rate'] = info.get('dividendRate')
        data['market_cap'] = info.get('marketCap')
        data['fifty_two_week_low'] = info.get('fiftyTwoWeekLow')
        data['fifty_two_week_high'] = info.get('fiftyTwoWeekHigh')
        
        # EV/EBITDA multiple (new user request)
        data['ev_ebitda'] = info.get('enterpriseToEbitda')
        
        # Format metrics
        if data['dividend_yield']:
            data['dividend_yield_pct'] = round(data['dividend_yield'] * 100, 2)
        else:
            data['dividend_yield_pct'] = None
            
        if data['market_cap']:
            data['market_cap_bn'] = round(data['market_cap'] / 1e9, 2)
        else:
            data['market_cap_bn'] = None

        # 3. Fetch yfinance stock news (we will fetch 5 and let Gemini filter)
        yf_news = []
        raw_news = ticker.news
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
        print(f"Error fetching yfinance data: {e}")
        data['error'] = str(e)
        
    return data

def fetch_regulatory_news():
    """
    Fetches global tobacco regulatory changes and policy updates using Google News RSS.
    We target FDA crackdowns, e-cigarette policies, and non-US developments.
    """
    queries = [
        "tobacco regulation FDA BTI",
        "illicit e-cigarette crackdown US nicotine",
        "tobacco excise tax vape ban Europe Asia"
    ]
    
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
            
            for item in root.findall(".//item")[:5]: # Limit to 5 per query
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
            
    return articles[:12]

def generate_comparison_chart(output_path):
    """
    Generates a 5-day performance chart comparing BTI, Altria (MO), and Philip Morris (PM).
    Stock performances are normalized to percent change starting from day 1's close.
    """
    print("Generating 5-day stock performance chart...")
    tickers = {"BTI": "BTI", "Altria (MO)": "MO", "Philip Morris (PM)": "PM"}
    
    plt.figure(figsize=(8, 4))
    # Set background styling to look premium (soft grey/white)
    plt.gcf().set_facecolor('#f8fafc')
    ax = plt.axes()
    ax.set_facecolor('#ffffff')
    
    colors = {"BTI": "#4f46e5", "Altria (MO)": "#f43f5e", "Philip Morris (PM)": "#0d9488"}
    linestyles = {"BTI": "-", "Altria (MO)": "--", "Philip Morris (PM)": "-."}
    widths = {"BTI": 2.5, "Altria (MO)": 1.5, "Philip Morris (PM)": 1.5}
    
    # We want exactly 5 trading days. We fetch 7 days to guarantee 5 trading days.
    for label, ticker_sym in tickers.items():
        ticker = yf.Ticker(ticker_sym)
        hist = ticker.history(period="7d")
        
        # Take the last 5 trading days
        if len(hist) > 5:
            hist = hist.tail(5)
            
        if hist.empty:
            print(f"Warning: No historical data found for {ticker_sym}")
            continue
            
        closes = hist['Close']
        base_price = closes.iloc[0]
        # Normalize to percent change starting at 0%
        pct_change = ((closes / base_price) - 1) * 100
        
        # Format the dates for x-axis (e.g. '06/12')
        dates = [d.strftime('%m/%d') for d in pct_change.index]
        
        plt.plot(dates, pct_change.values, 
                 label=label, 
                 color=colors[label], 
                 linestyle=linestyles[label], 
                 linewidth=widths[label],
                 marker='o' if label == "BTI" else None)
        
    plt.title("BTI vs. Altria & Philip Morris\n(Last 5 Trading Days Performance)", 
              fontsize=12, fontweight='bold', color='#1e293b', pad=10)
    plt.xlabel("Date", fontsize=10, color='#64748b')
    plt.ylabel("Performance Change (%)", fontsize=10, color='#64748b')
    
    # Format grid lines
    plt.grid(True, linestyle=':', alpha=0.6, color='#cbd5e1')
    
    # Add a horizontal line at 0% base
    plt.axhline(0, color='#94a3b8', linewidth=0.8, linestyle='-')
    
    # Customise legend and borders
    plt.legend(frameon=True, facecolor='#ffffff', edgecolor='#e2e8f0', loc='best')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cbd5e1')
    ax.spines['bottom'].set_color('#cbd5e1')
    ax.tick_params(axis='both', colors='#64748b')
    
    # Ensure layout fits neatly
    plt.tight_layout()
    
    # Save image
    plt.savefig(output_path, dpi=150, facecolor=plt.gcf().get_facecolor(), edgecolor='none')
    plt.close()
    print(f"Chart saved successfully to {output_path}")

if __name__ == "__main__":
    print("Fetching stock metrics...")
    data = fetch_bti_data()
    print(f"BTI EV/EBITDA: {data.get('ev_ebitda')}")
    
    # Test chart generation
    chart_file = os.path.join(os.path.dirname(__file__), "test_chart.png")
    generate_comparison_chart(chart_file)
