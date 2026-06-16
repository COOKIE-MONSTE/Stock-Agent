import agent
import data_fetcher
import email_service
import datetime
import os

def run_pipeline():
    """
    Runs the agent pipeline end-to-end:
    1. Generates the 5-day performance comparison chart.
    2. Runs BTI data collection and Gemini AI analysis.
    3. Sends the HTML email with the inline chart attached.
    """
    print("=" * 60)
    print("Starting BTI Stock Update Agent Pipeline")
    print(f"Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    chart_path = os.path.join(current_dir, "comparison_chart.png")
    local_report_path = os.path.join(current_dir, "last_report.html")
    
    try:
        # 1. Generate the comparison chart (Matplotlib)
        data_fetcher.generate_comparison_chart(chart_path)
        
        # 2. Run agent to get HTML content and BTI price details
        html_content, price, pct_change = agent.generate_bti_report()
        
        # Determine performance emoji
        emoji = "📈" if pct_change and pct_change >= 0 else "📉"
        change_sign = "+" if pct_change and pct_change > 0 else ""
        
        # 3. Formulate email subject
        subject = f"BTI Stock Update - ${price} ({change_sign}{pct_change}%){emoji}"
        
        print("\nPipeline execution complete. Formatting output...")
        
        # 4. Attempt to send email with inline chart attached
        sent = email_service.send_email(subject, html_content, chart_path)
        
        # 5. Output local preview logs
        print("\n" + "=" * 60)
        print("Pipeline Execution Summary:")
        print(f"  - BTI Price: ${price} ({change_sign}{pct_change}%)")
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

if __name__ == "__main__":
    run_pipeline()
