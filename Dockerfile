# Build this AFTER running `python main.py` locally, so that
# outputs/xgb_poisson_model.joblib exists to be copied in below.

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY outputs/xgb_poisson_model.joblib ./outputs/
COPY app.py .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
