#!/usr/bin/env python3
"""
Job Runner for Model Training Batch Job

This script runs the model training job at specified intervals.

Usage:
    python job_runner.py
"""

import os
import sys
import logging
import datetime as dt
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger

# Add the current directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from src.training_job import run_training_job

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """
    Main function to schedule and run the training job.
    
    By default, the job runs daily.
    The schedule can be configured using environment variables:
    - TRAINING_INTERVAL: interval in hours (e.g., 24 for daily)
    """
    try:
        logger.info("Starting job scheduler...")
        
        # Get interval from environment variable (default to 24 hours)
        interval_hours = int(os.environ.get('TRAINING_INTERVAL', '24'))
        logger.info(f"Training job will run every {interval_hours} hours")
        
        # Create scheduler
        scheduler = BlockingScheduler()
        
        # Add job to scheduler
        scheduler.add_job(
            run_training_job,
            trigger=IntervalTrigger(hours=interval_hours),
            max_instances=1,
            coalesce=True,
            id='model_training_job'
        )
        
        # Run job immediately
        logger.info("Running initial training job...")
        for job in scheduler.get_jobs():
            job.modify(next_run_time=dt.datetime.now())
        
        # Start scheduler
        logger.info("Starting scheduler...")
        scheduler.start()
        
    except (KeyboardInterrupt, SystemExit):
        logger.info("Job scheduler stopped")
    except Exception as e:
        logger.error(f"Fatal error in job_runner: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 