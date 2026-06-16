import time
import datetime
import subprocess
import os
import sys
import test_run

def run_loop(target_time_str="08:00"):
    """
    Runs a continuous background loop that checks the time.
    Triggers the report once daily at the target time.
    """
    print(f"Starting Stock Agent Scheduler Loop...")
    print(f"Target daily run time: {target_time_str}")
    print("Keep this terminal window open to keep the scheduler running.")
    print("-" * 50)
    
    last_run_date = None
    
    while True:
        now = datetime.datetime.now()
        current_time = now.strftime("%H:%M")
        current_date = now.date()
        
        # Check if the time matches and we haven't run yet today
        if current_time == target_time_str and last_run_date != current_date:
            print(f"\nTime matches target ({target_time_str}). Triggering pipeline...")
            test_run.run_pipeline()
            last_run_date = current_date
            print(f"Pipeline executed. Next run scheduled for tomorrow at {target_time_str}.")
            
        # Sleep for 30 seconds before checking again
        time.sleep(30)

def print_windows_scheduler_instructions():
    """
    Prints instructions for scheduling the agent using native Windows Task Scheduler.
    This is highly recommended since it runs in the background and starts on reboot.
    """
    script_path = os.path.abspath("test_run.py")
    python_exe = sys.executable
    
    print("=" * 70)
    print("RECOMMENDED: Schedule Native Windows Task Scheduler")
    print("=" * 70)
    print("To run this agent automatically every morning at 8:00 AM without keeping")
    print("this terminal open, you can create a Windows scheduled task.")
    print("\nRun the following command in PowerShell with Administrator privileges:")
    print("-" * 70)
    print(f'Register-ScheduledTask -TaskName "BTIStockAgent" -Trigger (New-ScheduledTaskTrigger -Daily -At "8:00am") -Action (New-ScheduledTaskAction -Execute "{python_exe}" -Argument "{script_path}" -WorkingDirectory "{os.path.dirname(script_path)}")')
    print("-" * 70)
    print("To disable or delete this task later, run:")
    print('Unregister-ScheduledTask -TaskName "BTIStockAgent" -Confirm:$false')
    print("=" * 70)

if __name__ == "__main__":
    print_windows_scheduler_instructions()
    print("\nDo you want to run the scheduler in this terminal window right now?")
    print("Press Ctrl+C at any time to exit.")
    print("-" * 50)
    try:
        run_loop("08:00")
    except KeyboardInterrupt:
        print("\nScheduler stopped.")
