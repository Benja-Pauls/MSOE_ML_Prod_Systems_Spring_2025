import logging
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger
from evaluation_job import run_job

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Create scheduler
    scheduler = BlockingScheduler()
    
    # Add job to scheduler
    scheduler.add_job(
        run_job,
        trigger=IntervalTrigger(minutes=15),
        max_instances=1,
        coalesce=True,
        id='evaluation_job'
    )
    
    # Run job immediately
    run_job()
    
    # Start scheduler
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        pass

if __name__ == "__main__":
    main()
