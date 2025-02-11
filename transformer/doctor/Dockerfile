FROM python:3.11-slim
WORKDIR /app
COPY services /app/services
COPY doctor /app/doctor
RUN pip install --no-cache-dir flask PyJWT flask-limiter python-dotenv gunicorn
RUN useradd -ms /bin/bash appuser && chown -R appuser:appuser /app
USER appuser
EXPOSE 5001
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5001", "doctor.app:app"]
