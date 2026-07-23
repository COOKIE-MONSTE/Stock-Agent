import agent
import data_fetcher
import email_service
import datetime
import os
import sys
from stocks_config import STOCKS

# Local marker file used by the once-per-day guard. Records the date of the
# last successful run ON THIS MACHINE. It deliberately does NOT travel with the
# repo (add it to .gitignore), and it's a no-op on ephemeral CI runners since
# the fresh checkout never has it -- so the GitHub Action always runs, while
# local double-fires (scheduler.py firing + you also running manually) are
# blocked. That double-fire is a real way to burn the tiny Gemini daily quota.
_RUN_MARKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".last_run_date")


def _already_ran_today():
    """True if a successful run has already been recorded for today's date."""
    try:
        with open(_RUN_MARKER, "r", encoding="utf-8") as f:
            return f.read().strip() == datetime.date.today().isoformat()
    except FileNotFoundError:
        return False
    except Exception:
        # If the marker is unreadable for any reason, don't block the run.
        return False


def _mark_ran_today():
    """Record that today's run completed, for the once-per-day guard."""
    try:
        with open(_RUN_MARKER, "w", encoding="utf-8") as f:
            f.write(datetime.date.today().isoformat())
    except Exception as e:
        print(f"Note: couldn't write run marker ({e}); once-per-day guard disabled this run.")


def run_pipeline(force=False):
    """
    Runs the agent pipeline end-to-end:
    1. Generates the 5-day performance comparison chart for the whole group.
    2. Runs data collection + Gemini analysis for every ticker in STOCKS.
    3. Sends the combined HTML email with the inline chart attached.

    once-per-day guard: if a run already completed today on this machine, the
    pipeline is skipped to protect the Gemini daily quota. Bypass with
    force=True, the --force CLI flag, or RUN_FORCE=1. The guard never fires on
    a clean CI checkout, so the scheduled GitHub Action is unaffected.
    """
    force = force or os.getenv("RUN_FORCE") == "1"
    if not force and _already_ran_today():
        print(f"Skip: pipeline already ran today ({datetime.date.today().isoformat()}) "
              f"on this machine. Use --force (or RUN_FORCE=1) to override.")
        print("This guard protects the limited Gemini daily quota from duplicate runs.")
        return

    print("=" * 60)
    print("Starting Portfolio Stock Update Agent Pipeline")
    print(f"Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    chart_path = os.path.join(current_dir, "comparison_chart.png")
    macro_chart_path = os.path.join(current_dir, "macro_chart.png")
    logo_path = os.path.join(current_dir, "assets", "logo.png")
    local_report_path = os.path.join(current_dir, "last_report.html")

    try:
        # 1. Generate the group comparison chart (Matplotlib)
        data_fetcher.generate_comparison_chart(chart_path, {t: t for t in STOCKS})

        # 2. Run agent to get the combined HTML report + per-ticker summary
        #    (this also generates the consumer/retail macro trend chart at
        #    macro_chart_path, from FRED data, if FRED_API_KEY is configured)
        html_content, summary = agent.generate_portfolio_report(macro_chart_path=macro_chart_path)

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

        # 5. Attempt to send email with inline charts attached
        sent = email_service.send_email(subject, html_content, chart_path, logo_path,
                                        macro_chart_path=macro_chart_path)

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

        # Record success LAST, so a failed run doesn't block a legitimate retry.
        _mark_ran_today()

    except Exception as e:
        print(f"\nPipeline execution failed: {e}")
        import traceback
        traceback.print_exc()
        raise  # re-raise so GitHub Actions actually marks the run as failed


if __name__ == "__main__":
    run_pipeline(force="--force" in sys.argv)
