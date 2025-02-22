import datetime as dt
from apscheduler.schedulers.blocking import BlockingScheduler
from cleaning_job import check_environment, run_job

def main():
    check_environment()
    scheduler = BlockingScheduler()
    scheduler.add_job(run_job, 'interval', minutes=15, max_instances=1, coalesce=True)

    for job in scheduler.get_jobs():
        job.modify(next_run_time=dt.datetime.now())

    scheduler.start()

if __name__ == "__main__":
    main()