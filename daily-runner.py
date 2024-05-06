import schedule
import time
import subprocess

def job():
    # Put your code here that you want to run every 24 hours
    # For example, you can run another Python script
    subprocess.run(["python", "daily-news-sql.py"])

# Schedule the job to run every 24 hours
schedule.every(24).hours.do(job)

while True:
    schedule.run_pending()
    time.sleep(1)