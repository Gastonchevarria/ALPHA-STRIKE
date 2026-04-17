FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create persistent directories
RUN mkdir -p learnings data/strategy_params

EXPOSE 9090

CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-9090}
