FROM python:3.11-slim

RUN useradd -m -u 1000 user
USER user

ENV PATH="/home/user/.local/bin:${PATH}"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Logging: console (Space Logs) + rotating file under ./logs (best-effort; override in HF UI if needed)
ENV LOG_LEVEL=INFO
ENV LOG_JSON=true
ENV LOG_FILE=./logs/app.log

WORKDIR /app

COPY --chown=user requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . /app

# Writable log directory for non-root user
USER root
RUN mkdir -p /app/logs && chown user:user /app/logs
USER user

EXPOSE 7860

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
