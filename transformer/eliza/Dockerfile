# Base image with Python
FROM python:3.11-slim
WORKDIR /app
# Copy service code and shared library
COPY services /app/services
COPY eliza /app/eliza
# Install dependencies
RUN pip install --no-cache-dir flask PyJWT flask-limiter python-dotenv gunicorn
# Switch to a non-root user for security
RUN useradd -ms /bin/bash appuser && chown -R appuser:appuser /app
USER appuser
# Expose service port
EXPOSE 5000
# Start the service with Gunicorn
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "eliza.app:app"]
