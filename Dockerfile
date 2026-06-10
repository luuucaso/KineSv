FROM python:3.10-slim

WORKDIR /app

COPY backend/requirements.txt .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY backend .

CMD uvicorn main:app --host 0.0.0.0 --port ${PORT}