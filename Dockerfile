FROM python:3.11

WORKDIR /app

# Install dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY env.py .
COPY modules/ modules/
COPY graders/ graders/
COPY tasks/ tasks/
COPY server/ server/
COPY email_triage_env/ email_triage_env/

ENV PORT=7860
EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
