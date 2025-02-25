FROM python:3.12-slim

WORKDIR /app
COPY . /app/

RUN pip3 install -U pip wheel
RUN pip3 install "psycopg[binary]" apscheduler marshmallow

CMD ["python3", "-u", "job_runner.py"]