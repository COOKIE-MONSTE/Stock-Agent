"""
Central config for the stock group covered by the daily report.

Each entry has:
  - display_name: fallback label if yfinance doesn't return one
  - catalyst_queries: 2-3 tailored Google News RSS search terms used to
    surface the kind of news that actually moves THIS company (regulatory,
    legal, demand/guidance, etc.) instead of generic headlines. This keeps
    the "exactly 3 curated angles" quality BTI had, per company, even
    though each company's angles are different.
"""

STOCKS = {
    "BTI": {
        "display_name": "British American Tobacco p.l.c. ADR",
        "catalyst_queries": [
            "tobacco regulation FDA BTI",
            "illicit e-cigarette crackdown US nicotine",
            "tobacco excise tax vape ban Europe Asia",
        ],
    },
    "ANF": {
        "display_name": "Abercrombie & Fitch Co.",
        "catalyst_queries": [
            "Abercrombie Fitch tariffs supply chain",
            "Abercrombie Fitch same store sales guidance",
            "apparel retail consumer spending trends",
        ],
    },
    "LYV": {
        "display_name": "Live Nation Entertainment",
        "catalyst_queries": [
            "Live Nation Ticketmaster antitrust lawsuit DOJ",
            "Live Nation concert touring revenue outlook",
            "ticketing fees legislation regulation",
        ],
    },
    "CHDN": {
        "display_name": "Churchill Downs Incorporated",
        "catalyst_queries": [
            "Churchill Downs gaming license regulation",
            "historical horse racing machines legal dispute",
            "sports betting expansion state regulation",
        ],
    },
    "NCLH": {
        "display_name": "Norwegian Cruise Line Holdings",
        "catalyst_queries": [
            "cruise line environmental fuel regulation",
            "Norwegian Cruise Line bookings demand outlook",
            "cruise industry capacity debt costs",
        ],
    },
    "DIS": {
        "display_name": "The Walt Disney Company",
        "catalyst_queries": [
            "Disney streaming subscriber growth profitability",
            "Disney parks attendance consumer spending",
            "ESPN sports media regulation antitrust",
        ],
    },
}
