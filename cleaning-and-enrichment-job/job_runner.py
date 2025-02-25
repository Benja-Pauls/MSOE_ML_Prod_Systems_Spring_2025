import datetime as dt
import sys
from apscheduler.schedulers.blocking import BlockingScheduler
from cleaning_job import check_environment, run_job

def main():
    try:
        print("Starting job scheduler...", flush=True)
        check_environment()
        
        scheduler = BlockingScheduler()
        scheduler.add_job(run_job, 'interval', minutes=15, max_instances=1, coalesce=True)

        print("Initial job scheduled, running immediately...", flush=True)
        for job in scheduler.get_jobs():
            job.modify(next_run_time=dt.datetime.now())

        print("Starting scheduler...", flush=True)
        scheduler.start()
    except Exception as e:
        print(f"Fatal error in job_runner: {str(e)}", flush=True)
        sys.exit(1)

if __name__ == "__main__":
    main()