FROM python:3.10-slim
WORKDIR /app
RUN pip install --no-cache-dir pandas flask
CMD ["python", "/app/create_sched_api.py"]