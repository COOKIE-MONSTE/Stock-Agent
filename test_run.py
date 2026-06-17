import agent
import data_fetcher
import email_service
import datetime
import os
from stocks_config import STOCKS

def run_pipeline():
    """
    Runs the agent pipeline end-to-end:
    1. Generates the 5-day performance comparison chart for the whole group.
    2. Runs data collection + Gemini analysis for every ticker in STOCKS.
    3. Sends the combined HTML email with the inline chart attached.
    """
    print("=" * 60)
    print("Starting Portfolio Stock Update Agent Pipeline")
    print(f"Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    chart_path = os.path.join(current_dir, "comparison_chart.png")
    local_report_path = os.path.join(current_dir, "last_report.html")

    try:
        # 1. Generate the group comparison chart (Matplotlib)
        data_fetcher.generate_comparison_chart(chart_path, {t: t for t in STOCKS})

        # 2. Run agent to get the combined HTML report + per-ticker summary
        html_content, summary = agent.generate_portfolio_report()

        # 3. Build subject line from the day's biggest mover
        movers = [s for s in summary if s.get("pct_change") is not None]
        if movers:
            biggest = max(movers, key=lambda s: abs(s["pct_change"]))
            emoji = "📈" if biggest["pct_change"] >= 0 else "📉"
            sign = "+" if biggest["pct_change"] > 0 else ""
            subject = (f"Portfolio Update - Biggest mover: {biggest['ticker']} "
                       f"({sign}{biggest['pct_change']}%) {emoji}")
        else:
            subject = "Portfolio Update"

        print("\nPipeline execution complete. Formatting output...")

        # 4. Save the local HTML copy (this previously never happened --
        #    the old version logged "Local HTML Saved" without writing it)
        with open(local_report_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        # 5. Attempt to send email with inline chart attached
        sent = email_service.send_email(subject, html_content, chart_path)

        # 6. Output local preview logs
        print("\n" + "=" * 60)
        print("Pipeline Execution Summary:")
        for s in summary:
            price_str = f"${s['price']}" if s['price'] is not None else "N/A"
            pct_str = f"{s['pct_change']}%" if s['pct_change'] is not None else "N/A"
            print(f"  - {s['ticker']}: {price_str} ({pct_str})")
        print(f"  - Chart Generated: {chart_path}")
        print(f"  - Local HTML Saved: {local_report_path}")
        if sent:
            print("  - Email Delivery: SUCCESSFUL")
        else:
            print("  - Email Delivery: SKIPPED (No valid SMTP settings found in .env)")
            print("    *Tip: You can open and view the generated report locally by double-clicking on 'last_report.html'.")
        print("=" * 60)

    except Exception as e:
        print(f"\nPipeline execution failed: {e}")
        import traceback
        traceback.print_exc()
        raise  # re-raise so GitHub Actions actually marks the run as failed

if __name__ == "__main__":
    run_pipeline()
