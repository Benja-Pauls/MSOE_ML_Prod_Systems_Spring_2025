FROM python:3.12-slim
WORKDIR /app
COPY . /app/
RUN pip3 install -U pip wheel
RUN pip3 install "psycopg[binary]" flask marshmallow scikit-learn==1.4.0 pandas numpy dill
EXPOSE 8888
CMD ["python3", "prediction_service.py", "--model-file", "patched_model.pkl"]