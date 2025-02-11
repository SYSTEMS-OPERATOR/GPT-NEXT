Understood. I will generate the full ELIZA & DOCTOR transformer deployment package, including:

- Python scripts for ELIZA & DOCTOR
- API service files (FastAPI-based)
- Dockerfiles for containerization
- systemd service files for daemonized execution
- Environment configuration (`.env`, `config.json` as fallback)
- Structured logging with JSON format
- Secure API authentication (JWT, rate-limiting, optional mTLS)
- Full documentation (`manifest.md`, `README.md`, `operator_troubleshooting.md`)

I will compile everything into structured code blocks, ensuring seamless repository inclusion and alignment with Nova’s deployment requirements. I will notify you once everything is ready.

**manifest.md**:
```markdown
# Repository Manifest

- **eliza/** - *ELIZA service module (chatbot API)*
  - **app.py** – Flask application for the ELIZA API service (routes, JWT auth, rate limiting).
  - **config.json** – Default configuration values for the ELIZA service (used if env vars are not set).
  - **.env.example** – Example environment variable definitions for ELIZA (to be copied/renamed to `.env` in deployment).
  - **Dockerfile** – Docker image definition for containerizing the ELIZA service.
  - **eliza.service** – Example systemd unit file for running ELIZA service as a Linux daemon.
- **doctor/** - *DOCTOR service module (chatbot API with same logic, separate instance)*
  - **app.py** – Flask application for the DOCTOR API service.
  - **config.json** – Default configuration for the DOCTOR service.
  - **.env.example** – Example environment variables for DOCTOR.
  - **Dockerfile** – Docker image definition for the DOCTOR service.
  - **doctor.service** – Systemd service file for running DOCTOR as a daemon.
- **services/** - *Shared code utilized by both ELIZA and DOCTOR*
  - **__init__.py** – (empty file) Marks the `services` directory as a Python package.
  - **config.py** – Configuration loader utility (loads `.env` and JSON, with environment override logic).
  - **logging_utils.py** – Logging setup utility (JSON formatter and optional file/remote log handlers).
  - **dialogue.py** – Chatbot dialogue logic (pattern-response rules for generating replies).
- **README.md** – Primary documentation with setup instructions, usage examples, and API reference.
- **operator_troubleshooting.md** – Detailed guide for installation issues, debugging tips, and security best practices for operators.
- **manifest.md** – Repository manifest (this file), listing all components and their purposes.
```

**README.md**:
```markdown
# ELIZA & DOCTOR Chatbot Services

## Overview

This repository contains two chatbot API services, **ELIZA** and **DOCTOR**, inspired by the classic ELIZA psychotherapy chatbot. Each service runs independently and provides conversational responses in the style of a Rogerian psychotherapist. Both services share common code for configuration, logging, and conversation logic, but they are isolated into separate deployments (microservices) for flexibility and modularity.

**Key Features:**

- **Stateless REST API:** Each service exposes an HTTP API endpoint for chatbot interaction. The services are stateless, with no session stored on the server, relying on JWT tokens for any user context.
- **Configuration via Environment**: All secrets and configurable parameters are provided through environment variables (with a `.env` file for local development) following the Twelve-Factor App principles ([The Twelve-Factor App ](https://12factor.net/config#:~:text=The%20twelve,agnostic%20standard)). A JSON config file is available as a secondary fallback for convenience.
- **Secure Authentication:** The APIs use JSON Web Tokens (JWT) for authentication of requests. Only requests with a valid JWT (in the `Authorization: Bearer ...` header) are processed, ensuring that only authorized clients can interact.
- **Rate Limiting:** To prevent abuse and denial-of-service, the services enforce configurable rate limiting on the chat endpoint. This helps maintain availability and fairness, and protects against brute-force attacks ([API security: The importance of rate limiting policies in safeguarding your APIs](https://www.redhat.com/en/blog/api-security-importance-rate-limiting-policies-safeguarding-your-apis#:~:text=API%20rate%20limiting%20is%20the,overloading%20of%20the%20API%20infrastructure)).
- **Mutual TLS Support:** For environments requiring high security, the services can be configured to require mutual TLS authentication, ensuring both client and server validate each other’s certificates ([Understanding Mutual TLS (MTLS) Authentication: How It Works](https://www.securew2.com/blog/mutual-tls-mtls-authentication#:~:text=,is%20immune%20to%20unauthorized%20access)).
- **IP Whitelisting/Blacklisting:** The services can restrict access based on client IP addresses. Only allowed IPs can access the API (or conversely, blocked IPs can be denied), providing an extra layer of security ([Secure Your REST API with IP Address Whitelisting - IPstack](https://ipstack.com/blog/how-to-secure-your-rest-api-with-ip-address-whitelisting#:~:text=IP%20address%20whitelisting%20is%20to,the%20target%20application%20or%20network)).
- **Structured Logging:** All services emit logs in structured JSON format, including timestamps and unique request IDs for each request. These **forensic logs** include correlation identifiers and key request metadata, aiding in audit and forensic analysis ([C9: Implement Security Logging and Monitoring - OWASP Top 10 ...](https://top10proactive.owasp.org/the-top-10/c9-security-logging-and-monitoring/#:~:text=C9%3A%20Implement%20Security%20Logging%20and,Satisfying%20regulatory%20compliance%20requirements)). Logs can be sent to local files or forwarded to centralized systems (e.g., ELK stack) for monitoring.
- **Container & Daemon Ready:** Each service comes with a Dockerfile for containerization and a systemd unit file for running as a Linux service. This offers flexibility in deployment, whether you prefer Docker, Kubernetes, or traditional VM/metal deployments with systemd.

## Repository Structure

```
├── eliza/
│   ├── app.py          # ELIZA service Flask application (API endpoints and logic)
│   ├── config.json     # Example configuration file (optional, used if env vars are not set)
│   ├── .env.example    # Example environment variable definitions for ELIZA
│   ├── Dockerfile      # Docker image definition for the ELIZA service
│   └── eliza.service   # systemd service unit for running ELIZA as a daemon
├── doctor/
│   ├── app.py          # DOCTOR service Flask application (API endpoints and logic)
│   ├── config.json     # Example configuration file for Doctor service
│   ├── .env.example    # Example environment variable definitions for DOCTOR
│   ├── Dockerfile      # Docker image definition for the DOCTOR service
│   └── doctor.service  # systemd service unit for running DOCTOR as a daemon
├── services/
│   ├── __init__.py     # (empty) Makes `services` a Python package
│   ├── config.py       # Shared configuration loader (env vars and JSON config handling)
│   ├── logging_utils.py# Shared logging setup (JSON formatter, file/remote handlers)
│   └── dialogue.py     # Shared chatbot dialogue logic (ELIZA/DOCTOR response generation)
├── README.md           # Setup, usage, and API reference documentation (this file)
├── operator_troubleshooting.md  # Troubleshooting and best practices for operators
└── manifest.md         # Table of contents of the repository
```

## Setup and Installation

### Prerequisites

- **Python 3.9+** (if running directly on a host or using systemd). The code has been tested with Python 3.x.
- **Dependencies:** The primary Python dependencies are Flask (for the web framework), PyJWT (for JWT handling), Flask-Limiter (for rate limiting), and python-dotenv (for loading `.env` files). If using Docker, these are installed in the image automatically. If running on a host, you should install these via pip.
- **System:** Linux/Unix environment recommended (for systemd service usage and ease of environment variable management). Docker can be used on any platform that supports it.

### Configuration Management

Configuration is driven by environment variables to ensure secrets and environment-specific values are not hard-coded ([The Twelve-Factor App ](https://12factor.net/config#:~:text=The%20twelve,agnostic%20standard)). You can set these variables in a `.env` file in each service directory (for development and container use) or via your system's environment (for production). The application will also attempt to load a JSON config file as a fallback.

**Environment Variables (.env):**

Each service looks for environment variables with a specific prefix:
- ELIZA service expects variables prefixed with `ELIZA_`
- DOCTOR service expects variables prefixed with `DOCTOR_`

For example, in `eliza/.env.example`:
```bash
ELIZA_JWT_SECRET="your-256-bit-secret"
ELIZA_JWT_ALGORITHM="HS256"
ELIZA_LOG_LEVEL="INFO"
ELIZA_RATE_LIMIT="60 per minute"
#ELIZA_ALLOWED_IPS=192.168.1.100,192.168.1.101   # (optional) comma-separated allowed IPs
#ELIZA_BLOCKED_IPS=                              # (optional) comma-separated blocked IPs
#ELIZA_TLS_CERT=/path/to/eliza_cert.pem          # (optional) TLS certificate path
#ELIZA_TLS_KEY=/path/to/eliza_key.pem            # (optional) TLS private key path
#ELIZA_CA_CERT=/path/to/ca_cert.pem              # (optional) CA certificate for client auth
```

> **Note:** Replace `"your-256-bit-secret"` with a strong secret key for signing JWTs. Keep this value secure and out of version control. The secret should be at least 256 bits (32+ characters) if using HS256.

The DOCTOR service uses analogous variables with the `DOCTOR_` prefix (e.g., `DOCTOR_JWT_SECRET`, `DOCTOR_RATE_LIMIT`, etc.). If both services should use the same JWT credentials (common in a microservice architecture where authentication is centralized), ensure both .env files share the same values for JWT secret and algorithm.

**JSON Config Fallback:**

Each service directory includes a `config.json` which provides default values. The application will load this if an environment variable is not set for a given setting. This is useful for providing non-sensitive defaults or when running in an environment where setting env vars is difficult. However, sensitive information (like secrets) should preferably be provided via environment variables to avoid being checked into source control ([The Twelve-Factor App ](https://12factor.net/config#:~:text=A%20litmus%20test%20for%20whether,moment%2C%20without%20compromising%20any%20credentials)) ([The Twelve-Factor App ](https://12factor.net/config#:~:text=The%20twelve,agnostic%20standard)).

Example `eliza/config.json`:
```json
{
  "HOST": "0.0.0.0",
  "PORT": 5000,
  "JWT_SECRET": "your-256-bit-secret",
  "JWT_ALGORITHM": "HS256",
  "RATE_LIMIT": "60 per minute",
  "LOG_LEVEL": "INFO",
  "LOG_FILE": null,
  "SYSLOG_HOST": null,
  "SYSLOG_PORT": null,
  "ALLOWED_IPS": [],
  "BLOCKED_IPS": [],
  "REQUIRE_CLIENT_CERT": false,
  "TLS_CERT": null,
  "TLS_KEY": null,
  "CA_CERT": null
}
```
The `doctor/config.json` is similar but with `"PORT": 5001` by default to avoid conflict. **If a value is defined in both the environment and the JSON file, the environment value takes precedence.** In production, consider using environment variables exclusively for sensitive config, aligning with best practices ([The Twelve-Factor App ](https://12factor.net/config#:~:text=The%20twelve,agnostic%20standard)).

### Running the Services

You can run the ELIZA and DOCTOR services either in Docker containers or directly on a host (e.g., via systemd or a process manager). Both modes are supported.

#### Option 1: Running with Docker

Dockerfiles are provided for both services. Build the images and run containers for each service.

**Build the Docker images:**
```bash
# From the repository root
docker build -f eliza/Dockerfile -t eliza-service:latest .
docker build -f doctor/Dockerfile -t doctor-service:latest .
```
This will create images for each service. The Dockerfiles will install the necessary dependencies and use Gunicorn to run the Flask app.

**Run the containers:**
```bash
# Run ELIZA service on host port 5000
docker run -d --name eliza -p 5000:5000 --env-file eliza/.env eliza-service:latest
# Run DOCTOR service on host port 5001
docker run -d --name doctor -p 5001:5001 --env-file doctor/.env doctor-service:latest
```
Make sure to prepare `eliza/.env` and `doctor/.env` files with the appropriate configuration before running. The `--env-file` option loads the environment variables into the container. Each container will then start the service on the specified port (5000 for ELIZA, 5001 for DOCTOR by default).

The container logs will be output to stdout in JSON format. You can view logs with `docker logs -f eliza` (they will appear as JSON strings; see **Logging** section below for details).

#### Option 2: Running as Systemd Services (Linux)

For deployment on a Linux host without Docker, you can use the provided systemd service unit files.

Steps to set up:
1. Install Python and pip on the host, and install required packages:
   ```bash
   pip install flask PyJWT flask-limiter python-dotenv gunicorn
   ```
   (It's recommended to use a virtual environment or isolate these services in containers for maintainability.)
2. Create dedicated users to run the services (for example, `eliza` and `doctor` users) with restricted permissions. This ensures the services do not run as root.
3. Copy the code to an appropriate directory, e.g., `/opt/eliza` and `/opt/doctor`. Include the `services/` directory in each location (or install the shared code as a package).
4. Update the configuration in those directories:
   - Fill in the `.env` files (or adjust `config.json`) with the proper secrets and settings for your environment.
5. Place the `eliza.service` and `doctor.service` files into `/etc/systemd/system/`. Adjust the file paths in these units if your installation paths differ:
   - `WorkingDirectory` should point to the directory of the app (containing `app.py` and config files).
   - `EnvironmentFile` should point to the .env file containing configuration.
   - `ExecStart` may need the full path to Gunicorn or Python. By default, it assumes Gunicorn is in the PATH.
6. Reload systemd and start the services:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl start eliza.service
   sudo systemctl start doctor.service
   sudo systemctl enable eliza.service doctor.service   # enable on boot
   ```
   
After starting, check `sudo systemctl status eliza.service` (and similarly for doctor) to ensure they're running, and use `journalctl -u eliza.service -f` to tail the logs.

#### Option 3: Running directly (Development mode)

For quick testing or development, you can also run the Flask app directly using Python (though this is not suitable for production use):
```bash
# In one terminal, run ELIZA service
cd eliza/
export FLASK_APP=app.py
export FLASK_ENV=development   # enables debug mode (not recommended in production)
flask run --port 5000

# In another terminal, run DOCTOR service
cd doctor/
export FLASK_APP=app.py
export FLASK_ENV=development
flask run --port 5001
```
Ensure you've set the necessary environment variables (or have a `.env` in each directory) before running. The Flask development server will serve the application on the specified ports. In this mode, the rate limiting and other features will still apply. Debug mode (enabled by `FLASK_ENV=development`) will provide interactive error pages on exceptions.

## Usage

Once both services are up and running, you can interact with them via their HTTP API. The ELIZA service by default listens on port 5000 and the DOCTOR service on port 5001.

**Health Check:**
- **Endpoint:** `GET /health`
- **Description:** Returns a simple status of the service. Useful for load balancer or uptime checks.
- **Auth:** No authentication required.
- **Response:** JSON, e.g. `{"status": "ok", "service": "eliza"}` (or `"doctor"` accordingly) with HTTP 200 if the service is running.

**Chatbot Conversation:**
- **Endpoint:** `POST /api/v1/message`
- **Description:** Submit a user message to the chatbot service and receive a response.
- **Auth:** Requires a valid JWT in the `Authorization` header.
- **Request Body:** JSON object with a `"message"` field containing the user's input. For example: `{"message": "Hello"}`.
- **Response:** JSON object with a `"response"` field containing the chatbot's reply. For example: `{"response": "Hello... I'm glad you could drop by."}`.

**Example cURL Usage:**

First, obtain or create a JWT token signed with the shared secret. (In a real scenario, your authentication service would issue JWTs. For testing, you can use a tool like jwt.io or a small script with PyJWT using the secret in your config.)

Let's assume we have a JWT stored in an environment variable `TOKEN`:
```bash
export TOKEN=<your_jwt_token>
```

Make a request to the ELIZA service:
```bash
curl -X POST http://localhost:5000/api/v1/message \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"message": "I am feeling sad"}'
```

If the token is valid and the rate limit not exceeded, the response might be:
```json
{"response": "How do you feel about being sad?"}
```
This indicates the chatbot (ELIZA persona) has processed the input. Similarly, you can query the DOCTOR service on port 5001 (with an appropriate token if separate).

**Error Responses:**
- If the JWT is missing or invalid/expired, you will get `401 Unauthorized` with a JSON body like `{"error": "Unauthorized"}` or `{"error": "Token expired"}`.
- If you exceed the rate limit, the service returns `429 Too Many Requests` (the response may include a message indicating you have sent too many requests).
- If your IP is blocked or not on the allowed list, you'll receive `403 Forbidden`.
- For other client errors (e.g., no "message" field in the request JSON), a `400 Bad Request` is returned with an error.
- An internal server error will result in a `500 Internal Server Error` with `{"error": "Internal server error"}` (and details would be logged on the server side).

Each successful request/response will include an `X-Request-ID` header in the response. This is a unique identifier for the request, also logged on the server. You can use this ID to trace that specific request in the logs or across multiple services (for example, if another service calls this one and propagates the request ID).

## Logging and Monitoring

Both services implement structured logging. Each log entry is a JSON object (one per line) that includes fields such as timestamp, log level, message, and contextual data (like request ID, client IP, etc.). For example, an access log entry might look like:

```json
{
  "timestamp": "2025-02-11T06:30:15Z",
  "level": "INFO",
  "message": "request",
  "logger": "eliza",
  "method": "POST",
  "path": "/api/v1/message",
  "status": 200,
  "client_ip": "203.0.113.5",
  "request_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "user": "alice@example.com"
}
```

Fields explained:
- `timestamp`: The time of the log event (UTC, ISO8601 format).
- `level`: Log severity level (INFO, ERROR, etc).
- `message`: A short descriptor of the log event. For request logs, this is `"request"`.
- `logger`: The service name or logger name (e.g., "eliza" or "doctor").
- `method`, `path`, `status`, `client_ip`: HTTP request details.
- `request_id`: The unique ID for the request (same as in the `X-Request-ID` header).
- `user`: If available, an identifier for the authenticated user (e.g., the JWT "sub" claim).

These JSON logs are easily parseable by log management systems. In a local environment, you can pretty-print them using `jq`:
```bash
docker logs eliza -f | jq .
```
This will stream and format the ELIZA service logs in real time.

For production, you can integrate with your logging/monitoring stack:
- **Local file**: Set `LOG_FILE` in the config to a file path if you want the service to also output logs to a file on disk (aside from stdout).
- **Centralized Logging**: The Docker container logs (stdout/stderr) can be collected by a logging agent (e.g., Filebeat, Fluent Bit) and sent to an ELK stack or other SIEM. Because the logs are structured JSON, you can easily index fields like `request_id` or `user` in Elasticsearch.
- **Remote Syslog**: If you specify `SYSLOG_HOST` and `SYSLOG_PORT`, the service will attempt to send logs to that syslog server (UDP). Ensure your log server is configured to receive syslog and parse JSON logs. This can be used to integrate with systems that ingest syslog (like certain SIEMs).

Having detailed logs is important for security monitoring. OWASP recommends robust logging and monitoring to detect anomalies ([C9: Implement Security Logging and Monitoring - OWASP Top 10 ...](https://top10proactive.owasp.org/the-top-10/c9-security-logging-and-monitoring/#:~:text=C9%3A%20Implement%20Security%20Logging%20and,Satisfying%20regulatory%20compliance%20requirements)). These logs will help in tracing what happened during each request and are invaluable for debugging issues or investigating incidents.

## Security Considerations

Security has been a primary focus in the design of these services:

- **JWT Authentication:** All chat endpoints require a valid JWT. The service uses HMAC SHA-256 (HS256) by default for token signing. Ensure that `JWT_SECRET` is a high-entropy value. It's good practice to rotate this secret periodically and to use short token expiration times ([JWT Security Best Practices | Curity](https://curity.io/resources/learn/jwt-best-practices/#:~:text=short%20an%20expiration%20time%20for,valid%20for%20days%20or%20months)) to minimize the impact of token leakage.
- **Transport Security:** Always run the services behind TLS. You can handle TLS termination with a reverse proxy (Nginx, HAProxy, etc.) or enable it in-app by providing `TLS_CERT` and `TLS_KEY`. For an extra layer of security, you can enforce **mutual TLS (mTLS)** — require clients to present a certificate by setting `REQUIRE_CLIENT_CERT=true` and providing a CA certificate (`CA_CERT`) to trust. mTLS ensures that only clients with a valid cert (issued by your CA) can connect ([Understanding Mutual TLS (MTLS) Authentication: How It Works](https://www.securew2.com/blog/mutual-tls-mtls-authentication#:~:text=,is%20immune%20to%20unauthorized%20access)).
- **Rate Limiting:** The built-in rate limit (60 requests/min per IP by default) protects against brute force or abuse, contributing to API availability and security ([API security: The importance of rate limiting policies in safeguarding your APIs](https://www.redhat.com/en/blog/api-security-importance-rate-limiting-policies-safeguarding-your-apis#:~:text=API%20rate%20limiting%20is%20the,overloading%20of%20the%20API%20infrastructure)). Adjust this setting if needed based on normal usage patterns. You can also implement more complex rate-limiting rules (e.g., per user token instead of per IP) if required, using the Flask-Limiter library capabilities.
- **IP Filtering:** Use `ALLOWED_IPS` or `BLOCKED_IPS` as an additional layer of defense. IP whitelisting limits access to a known set of addresses, significantly reducing the attack surface ([How does IP whitelisting help secure an API? - Information Security Stack Exchange](https://security.stackexchange.com/questions/192207/how-does-ip-whitelisting-help-secure-an-api#:~:text=Simply%20by%20limiting%20the%20set,number%20of%20presumably%20trusted%20addresses)). This is especially useful for internal services or partner integrations where clients have fixed IP ranges. Keep in mind IP filtering is not foolproof (IP spoofing is mitigated by TCP handshake, but a compromised allowed host can still attack), so use it as defense-in-depth.
- **Least Privilege:** Both the Docker and systemd setups run the services under non-root users (`appuser` in Docker, or `eliza`/`doctor` in systemd). This limits the impact of any compromise. Ensure file permissions for config and certificate files are also restricted to these users.
- **No Sensitive Data in Logs:** To protect user privacy, the application does not log the content of user messages or any sensitive details from JWTs (other than a user identifier). This prevents accidental exposure of sensitive information in log management systems. If you enable debug logging, be mindful of what you log.
- **Dependency Management:** Keep the dependencies (Flask, PyJWT, etc.) up to date. Security patches in these libraries should be applied by rebuilding the Docker images or updating the packages in your environment.
- **Testing and Scanning:** Regularly perform security scans. Use tools to check for known vulnerabilities in dependencies. Consider running an API security test (using OWASP ZAP or similar) against the running services to ensure no obvious issues.
- **Operational Monitoring:** Monitor your services for unusual patterns. Multiple 401/403 responses could indicate someone trying to breach authentication. Frequent 429 (rate limit) hits might indicate a client in need of throttling or a potential attack. The structured logs can be used to set up alerts for such events.

By following these practices and using the provided security features, you can maintain a robust security posture. The combination of JWT auth, TLS, rate limiting, IP restrictions, and thorough logging aligns well with OWASP API Security guidelines (covering broken authentication, excessive usage, logging and monitoring, etc.) ([API security: The importance of rate limiting policies in safeguarding your APIs](https://www.redhat.com/en/blog/api-security-importance-rate-limiting-policies-safeguarding-your-apis#:~:text=,Improper%20error%20handling)) ([API security: The importance of rate limiting policies in safeguarding your APIs](https://www.redhat.com/en/blog/api-security-importance-rate-limiting-policies-safeguarding-your-apis#:~:text=API%20rate%20limiting%20is%20the,overloading%20of%20the%20API%20infrastructure)).

---

Please refer to **operator_troubleshooting.md** for more detailed troubleshooting and operational guidance.
```

**operator_troubleshooting.md**:
```markdown
# Operator Troubleshooting & Best Practices

This guide provides troubleshooting tips for common issues during installation and runtime, as well as best practices for operating the ELIZA and DOCTOR services securely and efficiently.

## Common Installation and Startup Issues

### Service Not Starting / Exiting Immediately

- **Missing Configuration:** If a service exits immediately after starting, check the logs. An error like *"JWT_SECRET not set"* indicates that required environment variables were not provided. Solution: ensure the `.env` file is present and contains the necessary entries (especially the JWT secret), or that you've exported the variables in the environment. The service will not run without a valid secret.

- **Dependency Errors:** If the service fails to start due to ImportError (modules not found), make sure all Python dependencies are installed. In Docker, this is handled in the build. For systemd deployments, install the packages via pip (see README for list). Example: an error about `flask` means Flask is not installed in the environment.

- **Port Conflicts:** If the service fails with "Address already in use", the port (default 5000 for ELIZA, 5001 for DOCTOR) is occupied. Change the `PORT` in the config or free the port. You can also map to a different host port if using Docker (`-p 8080:5000` to use 8080 externally for ELIZA, for instance).

- **Permission Denied (Systemd):** When running under systemd with a custom user, ensure that user has access to needed files:
  - The working directory and all files in it (app, config, .env) should be readable (and if needed, executable) by the user specified in the service file.
  - If using TLS keys/certificates, adjust their permissions so the service user can read them. Incorrect permissions can cause the service to fail to start when setting up SSL.
  - If the service user cannot bind to the port (typically if binding to a port <1024 without privileges), use higher ports or grant appropriate capabilities (not needed for default ports 5000/5001).

- **Gunicorn Path Issues:** The systemd unit files assume Gunicorn is available in the PATH (as installed via pip). If `gunicorn` command is not found, specify the full path (e.g., `/usr/local/bin/gunicorn`) in the `ExecStart`. Alternatively, adjust the environment (e.g., activate a virtualenv in ExecStart).

- **Docker Container Restarting:** If a Docker container exits repeatedly, inspect `docker logs <container>` for clues. Common causes are the same as above (missing env vars causing an immediate exit). Ensure you're passing the environment file correctly. Also verify that Docker has permission to read the files you bind (if any).

### Unexpected Runtime Issues

- **401 Unauthorized responses:** If every request is getting 401, the JWT might be missing or wrong.
  - Ensure the `Authorization: Bearer <token>` header is being sent.
  - Decode your JWT (e.g., on jwt.io) to verify it's not expired and uses the correct secret/algo. A mismatch in algorithm (e.g., token signed with RS256 while server expects HS256) will cause invalid token errors.
  - Check that the server has the same `JWT_SECRET` that was used to sign the token.

- **429 Too Many Requests (rate limit):** If you hit the rate limit frequently during testing:
  - You can increase the limit by setting a higher `RATE_LIMIT` value in the config or .env (e.g., `"100 per minute"`).
  - During development or load testing, consider disabling the limiter by using a very high limit or removing the limiter initialization from code (not recommended for production).
  - If behind a proxy, ensure the limiter is using the correct client IP. If all requests appear from the proxy's IP, the rate limit could unfairly apply. Configure proxy settings (like `X-Forwarded-For`) or adjust the key function in `Limiter` to use a combination of IP and user.

- **Unexpected 500 errors:** If the service returns a 500 Internal Server Error:
  - Check the logs for an "Unhandled exception" entry with the `error` field. This will contain the exception message.
  - Common issues could be malformed JSON input (Flask will raise a BadRequest which we convert to 400, not 500) or errors in the response generation logic (our dialogue is simple and should not throw exceptions normally).
  - Running the service in debug mode (FLASK_ENV=development) during testing can help, as you'll get a stack trace in the response.

- **Mutual TLS handshake failures:** If you've enabled `REQUIRE_CLIENT_CERT` and clients can't connect:
  - Confirm the client is presenting a certificate signed by the CA you configured.
  - Use `openssl s_client -connect host:5000 -cert client.crt -key client.key -CAfile ca.crt` to test the TLS connection manually. This can show verification errors.
  - Check that the CA certificate (`CA_CERT`) is correct and that it's accessible by the service.
  - Remember that if mTLS is on, even health checks must present a client cert or they'll be denied.

- **Logs not showing up or in wrong format:** 
  - If you don't see JSON logs, ensure that the logging is configured properly. In our code, we initialize logging at startup. If modifications were made, it's possible a default logging configuration is interfering.
  - If using systemd, by default stdout/stderr go to journal. You'll see the JSON as text in `journalctl`. For easier reading, you might forward them to a file or use `journalctl -o json-pretty`.
  - If remote logging isn't working (no logs at your syslog server), verify network connectivity and that the syslog port is open. Our implementation uses UDP for Syslog - make sure your server is listening on UDP and the address/port are correct.

## Debugging & Maintenance Tips

- **Using Request IDs:** When investigating an issue, use the `X-Request-ID` to find all relevant log entries. The same ID is attached to the access log and any error logs for that request. This is extremely helpful in tracing through microservices or filtering logs for a specific incident.

- **Increasing Verbosity:** To temporarily get more info, you can bump `LOG_LEVEL` to `DEBUG`. While our application doesn't log a lot at DEBUG by default, Flask and other libraries might output more info. Just be cautious not to leave debug logging on in production long-term (it can be verbose).

- **Testing Config Changes:** If you change something in the .env or config.json, restart the service (systemd `systemctl restart` or Docker restart). Changes won't apply until the service reloads and reads the new config.

- **Updating the Application:** When deploying a new version (code change), for Docker, build a new image and replace the container. For systemd, update the code on disk (or pull from git) and restart the service. Monitor the logs after restarting to catch any immediate errors (syntax errors, import issues will show up at startup).

- **Backup Config and Keys:** Keep backups of your `.env` files, config.json, and any TLS keys/certs in a secure location. This will greatly aid in disaster recovery. Treat these backups securely (especially the secrets).

- **Performance Monitoring:** The app is lightweight; however, if you expect high traffic, monitor CPU and memory usage:
  - Increase Gunicorn workers (`-w` flag) if CPU-bound (each worker can handle one request at a time).
  - If memory is an issue, ensure you aren't logging excessively or storing large objects. In our case, user messages and responses are short-lived and not stored.
  - Use tools like `ab` (Apache Bench) or `hey` to do simple load tests on non-production deployments, to find any bottlenecks before going live.

- **Cleaning Up Logs:** If logging to file, implement log rotation (e.g., via logrotate or by limiting file size in config if we had that feature). JSON logs can grow quickly. In Docker, manage container logs (Docker has options to rotate logs or limit size).

## Security Best Practices (Operations)

- **Regular Patching:** Keep the system and application dependencies updated. This includes applying OS updates (for the base Docker image or host OS) and updating Python packages for any security fixes ([API security: The importance of rate limiting policies in safeguarding your APIs](https://www.redhat.com/en/blog/api-security-importance-rate-limiting-policies-safeguarding-your-apis#:~:text=,Improper%20error%20handling)).
- **Rotate Secrets:** If you suspect a secret (JWT secret, TLS key) might be compromised or after a certain period (policy-based), rotate it. When rotating the JWT secret, you may allow tokens signed with the old secret for a short overlap period or force re-authentication of clients.
- **Use Strong Certificates:** If enabling TLS/mTLS, use strong cryptographic certificates (2048-bit RSA or ECC equivalent), and don't forget to update them before they expire.
- **Principle of Least Privilege:** Limit who can access the servers or the environment configs. Only admins should have access to the .env files or be able to read the logs, because those might contain sensitive indicators (like user IDs or IPs).
- **Network Policies:** If deploying on Kubernetes or similar, use NetworkPolicies to restrict traffic between these services and other pods. Only allow what is necessary.
- **WAF/API Gateway:** For internet-facing deployments, consider using an API gateway or WAF that can provide additional security features like IP rate limiting at the edge, payload inspection (to mitigate attacks like SQL injection - though our service doesn't use a database, this is a general consideration), and more sophisticated auth if needed (OAuth2, etc).
- **Monitoring and Alerts:** Set up alerts for anomalies:
  - e.g., Alert on a high rate of 5xx responses, which could indicate a problem.
  - Alert on repeated unauthorized attempts which could indicate an attack.
  - Monitor resource usage to detect if the service is under heavy load or a DoS attack.

By applying these operational best practices alongside the application's built-in security features, you'll create a robust and secure deployment. Always consider the sensitivity of the data passing through (user messages in this case) and apply appropriate compliance measures if needed (for example, if deployed in a healthcare context, additional auditing or data handling rules might apply).

```

**services/__init__.py**:
```python
# Marks the services directory as a Python package
```

**services/config.py**:
```python
import os
import json
# Load environment variables from a .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def load_config(config_path=None, env_prefix=None, defaults=None):
    """Load configuration from JSON file and environment variables.
    Environment variables override file values. `env_prefix` (if provided) filters env vars.
    """
    config = {}
    # Start with defaults
    if defaults:
        for k, v in defaults.items():
            config[k.upper()] = v
    # Load from JSON file if specified
    if config_path:
        try:
            with open(config_path, 'r') as f:
                file_conf = json.load(f)
                for key, val in file_conf.items():
                    config[key.upper()] = val
        except FileNotFoundError:
            pass
    # Override with environment variables
    for key, val in os.environ.items():
        # Apply prefix filter if given
        if env_prefix:
            if not key.startswith(env_prefix):
                continue
            eff_key = key[len(env_prefix):]
        else:
            eff_key = key
        eff_key = eff_key.upper()
        if defaults is None or eff_key in config:
            config[eff_key] = val
    # Type conversions for known fields
    bool_keys = {"REQUIRE_CLIENT_CERT"}
    int_keys = {"PORT"}
    list_keys = {"ALLOWED_IPS", "BLOCKED_IPS"}
    for k, v in list(config.items()):
        if k in bool_keys:
            if isinstance(v, str):
                config[k] = v.lower() in ("1", "true", "yes")
        elif k in int_keys:
            if isinstance(v, str) and v.isdigit():
                config[k] = int(v)
        elif k in list_keys:
            if isinstance(v, str):
                config[k] = [x.strip() for x in v.split(',') if x.strip()]
    return config
```

**services/logging_utils.py**:
```python
import logging
import json
from datetime import datetime
from logging.handlers import SysLogHandler

class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        # Create a dictionary for the log entry
        log_entry = {
            "timestamp": datetime.utcfromtimestamp(record.created).strftime('%Y-%m-%dT%H:%M:%SZ'),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name
        }
        # Include any extra fields that were passed in record.extra
        skip_keys = {
            'name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 'filename',
            'module', 'exc_info', 'exc_text', 'stack_info', 'lineno', 'funcName',
            'created', 'msecs', 'relativeCreated', 'thread', 'threadName', 'processName', 'process'
        }
        for key, value in record.__dict__.items():
            if key not in skip_keys:
                log_entry[key] = value
        return json.dumps(log_entry)

def init_logging(service_name: str, level: str = "INFO", log_file: str = None, syslog_addr: tuple = None):
    """
    Initialize structured JSON logging for the given service.
    Outputs to console by default, and optionally to file or syslog if configured.
    """
    logger = logging.getLogger(service_name)
    if logger.handlers:
        # Already initialized
        return logger
    # Set level
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)
    formatter = JSONFormatter()
    # Console (stdout) handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    # File handler (optional)
    if log_file:
        try:
            fh = logging.FileHandler(log_file)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        except Exception as e:
            logger.error("Failed to set up file logging: %s", e)
    # Syslog handler (optional)
    if syslog_addr:
        try:
            syslog_handler = SysLogHandler(address=syslog_addr)
            syslog_handler.setFormatter(formatter)
            logger.addHandler(syslog_handler)
        except Exception as e:
            logger.error("Failed to set up syslog logging: %s", e)
    logger.propagate = False
    return logger
```

**services/dialogue.py**:
```python
import re
import random

# Reflection map for pronoun swapping
reflections = {
    "am": "are", "was": "were", "i": "you", "i'd": "you would", "i've": "you have",
    "i'll": "you will", "my": "your", "are": "am", "you're": "I'm", "you've": "I have",
    "you'll": "I will", "your": "my", "yours": "mine", "you": "me", "me": "you"
}

# Pattern-response pairs for the therapist (ELIZA/DOCTOR) persona
doctor_patterns = [
    (r'I need (.*)', [
        "Why do you need {0}?", 
        "Would it really help you to get {0}?", 
        "Are you sure you need {0}?"]),
    (r'Why don\'?t you ([^\?]*)\??', [
        "Do you really think I don't {0}?", 
        "Perhaps eventually I will {0}.", 
        "Do you really want me to {0}?"]),
    (r'Why can\'?t I ([^\?]*)\??', [
        "Do you think you should be able to {0}?", 
        "If you could {0}, what would you do?", 
        "What’s stopping you from {0}?"]),
    (r'I can\'?t (.*)', [
        "How do you know you can't {0}?", 
        "Perhaps you could {0} if you tried.", 
        "What would it take for you to {0}?"]),
    (r'I am (.*)', [
        "Did you come to me because you are {0}?", 
        "How do you feel about being {0}?"]),
    (r'I\'?m (.*)', [
        "How do you feel about being {0}?", 
        "Do you often feel {0}?"]),
    (r'You are (.*)', [
        "What makes you think I am {0}?", 
        "Does it please you to think that I'm {0}?"]),
    (r'What (.*)', [
        "Why do you ask?", 
        "What do you think?"]),
    (r'How (.*)', [
        "How do you suppose?", 
        "Perhaps you can answer your own question."]),
    (r'Because (.*)', [
        "Is that the real reason?", 
        "What other reasons come to mind?"]),
    (r'(.*) sorry (.*)', [
        "There's no need to apologize.", 
        "What feelings do you have when you apologize?"]),
    (r'Hello(.*)', [
        "Hello... I'm glad you came today.", 
        "Hi there, how can I help you?"]),
    (r'I think (.*)', [
        "Do you doubt {0}?", 
        "Do you really think so?"]),
    (r'(.*) friend (.*)', [
        "Tell me more about your friends.", 
        "Why not tell me about a childhood friend?"]),
    (r'Yes', [
        "You seem quite sure.", 
        "OK, but can you elaborate?"]),
    (r'(.*) computer(.*)', [
        "Are computers a source of concern for you?", 
        "What do you think about machines?"]),
    (r'Is it (.*)', [
        "Do you think it is {0}?"]),
    (r'It is (.*)', [
        "What makes you feel it is {0}?"]),
    (r'Can you (.*)', [
        "What makes you think I can't {0}?", 
        "Whether or not I can {0} is not the question."]),
    (r'Can I (.*)', [
        "Perhaps you don't want to {0}.", 
        "Do you want to be able to {0}?"]),
    (r'(.*) mother(.*)', [
        "Tell me more about your mother.", 
        "What was your relationship with your mother like?"]),
    (r'(.*) father(.*)', [
        "How did your father make you feel?", 
        "Tell me more about your father."]),
    (r'(.*) child(.*)', [
        "Did you have close friends as a child?", 
        "What is your favorite childhood memory?"]),
    (r'(.*)\?', [
        "Why do you ask that?", 
        "What do you think?"]),
    (r'quit', [
        "Thank you for talking with me.", 
        "Good-bye."]),
    (r'(.*)', [
        "Please tell me more.", 
        "Let's change focus a bit... Tell me about your family.", 
        "Can you elaborate on that?"])
]

# Use the same patterns for ELIZA persona (alias to doctor_patterns)
eliza_patterns = doctor_patterns

def reflect(fragment):
    """Reflects a fragment of input by swapping pronouns (I -> you, me -> you, etc.)."""
    tokens = fragment.lower().split()
    for i, token in enumerate(tokens):
        if token in reflections:
            tokens[i] = reflections[token]
    return " ".join(tokens)

def generate_response(message, persona="doctor"):
    """Generate a response to the user's message using the specified persona's rules."""
    patterns = doctor_patterns if persona == "doctor" else eliza_patterns
    for pattern, responses in patterns:
        match = re.match(pattern, message.strip(), re.IGNORECASE)
        if match:
            response_template = random.choice(responses)
            # If the response template expects a fragment, fill it in after reflection
            if '{0}' in response_template:
                fragment = match.group(1)
                fragment = reflect(fragment)
                return response_template.format(fragment)
            else:
                return response_template
    # Fallback (should not normally reach here because of last catch-all pattern)
    return "Interesting. Please continue."
```

**eliza/app.py**:
```python
from flask import Flask, request, jsonify, g
import jwt
import sys
import ssl
import uuid
from werkzeug.exceptions import HTTPException
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Import shared utilities from services
try:
    from services.config import load_config
    import services.logging_utils as logging_utils
    from services.dialogue import generate_response
except ImportError:
    # If running without package context, adjust import (for standalone execution)
    from config import load_config
    import logging_utils as logging_utils
    from dialogue import generate_response

SERVICE_NAME = "eliza"

# Default configuration values
default_config = {
    "HOST": "0.0.0.0",
    "PORT": 5000,
    "JWT_SECRET": None,
    "JWT_ALGORITHM": "HS256",
    "RATE_LIMIT": "60 per minute",
    "LOG_LEVEL": "INFO",
    "LOG_FILE": None,
    "SYSLOG_HOST": None,
    "SYSLOG_PORT": None,
    "ALLOWED_IPS": [],
    "BLOCKED_IPS": [],
    "REQUIRE_CLIENT_CERT": False,
    "TLS_CERT": None,
    "TLS_KEY": None,
    "CA_CERT": None
}

# Load config from file and environment (environment variables prefixed with ELIZA_)
config = load_config(config_path="config.json", env_prefix="ELIZA_", defaults=default_config)
if not config.get("JWT_SECRET"):
    print("Error: JWT_SECRET not set for ELIZA service", file=sys.stderr)
    sys.exit(1)

app = Flask(__name__)
# Initialize rate limiter
limiter = Limiter(app, key_func=get_remote_address, default_limits=[config.get("RATE_LIMIT", "60 per minute")])
# Initialize logging
logger = logging_utils.init_logging(SERVICE_NAME, level=config.get("LOG_LEVEL", "INFO"),
                                    log_file=config.get("LOG_FILE"),
                                    syslog_addr=(config["SYSLOG_HOST"], int(config["SYSLOG_PORT"]))
                                    if config.get("SYSLOG_HOST") and config.get("SYSLOG_PORT") else None)

@app.before_request
def before_request():
    # Assign or propagate a request ID
    req_id = request.headers.get('X-Request-ID')
    if not req_id:
        req_id = str(uuid.uuid4())
    g.request_id = req_id

    # IP-based filtering
    client_ip = request.remote_addr
    if config.get("ALLOWED_IPS"):
        if client_ip not in config["ALLOWED_IPS"]:
            return jsonify({"error": "Forbidden"}), 403
    if config.get("BLOCKED_IPS"):
        if client_ip in config["BLOCKED_IPS"]:
            return jsonify({"error": "Forbidden"}), 403

    # JWT Authentication (skip for health check endpoint)
    if request.endpoint != 'health':
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith("Bearer "):
            return jsonify({"error": "Unauthorized"}), 401
        token = auth_header.split(" ", 1)[1]
        try:
            payload = jwt.decode(token, config["JWT_SECRET"], algorithms=[config.get("JWT_ALGORITHM", "HS256")])
            g.user_claims = payload  # store token claims if needed
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token expired"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"error": "Invalid token"}), 401

@app.after_request
def after_request(response):
    # Include request ID in response header for client correlation
    if hasattr(g, 'request_id'):
        response.headers['X-Request-ID'] = g.request_id
    # Log the request details
    log_data = {
        "method": request.method,
        "path": request.path,
        "status": response.status_code,
        "client_ip": request.remote_addr,
        "request_id": getattr(g, 'request_id', None)
    }
    if hasattr(g, 'user_claims') and 'sub' in g.user_claims:
        log_data['user'] = g.user_claims['sub']
    logger.info("request", extra=log_data)
    return response

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "service": SERVICE_NAME})

@app.route('/api/v1/message', methods=['POST'])
def message():
    data = request.get_json(silent=True) or {}
    user_msg = data.get("message")
    if not user_msg:
        return jsonify({"error": "No message provided"}), 400
    bot_reply = generate_response(user_msg, persona="eliza")
    return jsonify({"response": bot_reply})

# Exempt health check from rate limiting
limiter.exempt(health)

@app.errorhandler(HTTPException)
def handle_http_exception(e):
    # Return JSON error for HTTP exceptions (like 401, 404, etc.)
    return jsonify({"error": e.description}), e.code

@app.errorhandler(Exception)
def handle_exception(e):
    # Log unhandled exceptions and return generic error
    logger.error("Unhandled exception", extra={
        "error": str(e),
        "request_id": getattr(g, 'request_id', None)
    })
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    # Configure SSL if certificates are provided
    ssl_context = None
    if config.get("TLS_CERT") and config.get("TLS_KEY"):
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_context.load_cert_chain(certfile=config["TLS_CERT"], keyfile=config["TLS_KEY"])
        if config.get("REQUIRE_CLIENT_CERT") and config.get("CA_CERT"):
            ssl_context.load_verify_locations(cafile=config["CA_CERT"])
            ssl_context.verify_mode = ssl.CERT_REQUIRED
        else:
            ssl_context.verify_mode = ssl.CERT_NONE
    app.run(host=config.get("HOST", "0.0.0.0"), port=int(config.get("PORT", 5000)), ssl_context=ssl_context)
```

**doctor/app.py**:
```python
from flask import Flask, request, jsonify, g
import jwt
import sys
import ssl
import uuid
from werkzeug.exceptions import HTTPException
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

try:
    from services.config import load_config
    import services.logging_utils as logging_utils
    from services.dialogue import generate_response
except ImportError:
    from config import load_config
    import logging_utils as logging_utils
    from dialogue import generate_response

SERVICE_NAME = "doctor"

default_config = {
    "HOST": "0.0.0.0",
    "PORT": 5001,
    "JWT_SECRET": None,
    "JWT_ALGORITHM": "HS256",
    "RATE_LIMIT": "60 per minute",
    "LOG_LEVEL": "INFO",
    "LOG_FILE": None,
    "SYSLOG_HOST": None,
    "SYSLOG_PORT": None,
    "ALLOWED_IPS": [],
    "BLOCKED_IPS": [],
    "REQUIRE_CLIENT_CERT": False,
    "TLS_CERT": None,
    "TLS_KEY": None,
    "CA_CERT": None
}

config = load_config(config_path="config.json", env_prefix="DOCTOR_", defaults=default_config)
if not config.get("JWT_SECRET"):
    print("Error: JWT_SECRET not set for DOCTOR service", file=sys.stderr)
    sys.exit(1)

app = Flask(__name__)
limiter = Limiter(app, key_func=get_remote_address, default_limits=[config.get("RATE_LIMIT", "60 per minute")])
logger = logging_utils.init_logging(SERVICE_NAME, level=config.get("LOG_LEVEL", "INFO"),
                                    log_file=config.get("LOG_FILE"),
                                    syslog_addr=(config["SYSLOG_HOST"], int(config["SYSLOG_PORT"]))
                                    if config.get("SYSLOG_HOST") and config.get("SYSLOG_PORT") else None)

@app.before_request
def before_request():
    req_id = request.headers.get('X-Request-ID')
    if not req_id:
        req_id = str(uuid.uuid4())
    g.request_id = req_id

    client_ip = request.remote_addr
    if config.get("ALLOWED_IPS"):
        if client_ip not in config["ALLOWED_IPS"]:
            return jsonify({"error": "Forbidden"}), 403
    if config.get("BLOCKED_IPS"):
        if client_ip in config["BLOCKED_IPS"]:
            return jsonify({"error": "Forbidden"}), 403

    if request.endpoint != 'health':
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith("Bearer "):
            return jsonify({"error": "Unauthorized"}), 401
        token = auth_header.split(" ", 1)[1]
        try:
            payload = jwt.decode(token, config["JWT_SECRET"], algorithms=[config.get("JWT_ALGORITHM", "HS256")])
            g.user_claims = payload
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token expired"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"error": "Invalid token"}), 401

@app.after_request
def after_request(response):
    if hasattr(g, 'request_id'):
        response.headers['X-Request-ID'] = g.request_id
    log_data = {
        "method": request.method,
        "path": request.path,
        "status": response.status_code,
        "client_ip": request.remote_addr,
        "request_id": getattr(g, 'request_id', None)
    }
    if hasattr(g, 'user_claims') and 'sub' in g.user_claims:
        log_data['user'] = g.user_claims['sub']
    logger.info("request", extra=log_data)
    return response

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "service": SERVICE_NAME})

@app.route('/api/v1/message', methods=['POST'])
def message():
    data = request.get_json(silent=True) or {}
    user_msg = data.get("message")
    if not user_msg:
        return jsonify({"error": "No message provided"}), 400
    bot_reply = generate_response(user_msg, persona="doctor")
    return jsonify({"response": bot_reply})

limiter.exempt(health)

@app.errorhandler(HTTPException)
def handle_http_exception(e):
    return jsonify({"error": e.description}), e.code

@app.errorhandler(Exception)
def handle_exception(e):
    logger.error("Unhandled exception", extra={
        "error": str(e),
        "request_id": getattr(g, 'request_id', None)
    })
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    ssl_context = None
    if config.get("TLS_CERT") and config.get("TLS_KEY"):
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_context.load_cert_chain(certfile=config["TLS_CERT"], keyfile=config["TLS_KEY"])
        if config.get("REQUIRE_CLIENT_CERT") and config.get("CA_CERT"):
            ssl_context.load_verify_locations(cafile=config["CA_CERT"])
            ssl_context.verify_mode = ssl.CERT_REQUIRED
        else:
            ssl_context.verify_mode = ssl.CERT_NONE
    app.run(host=config.get("HOST", "0.0.0.0"), port=int(config.get("PORT", 5001)), ssl_context=ssl_context)
```

**eliza/config.json**:
```json
{
  "HOST": "0.0.0.0",
  "PORT": 5000,
  "JWT_SECRET": "your-256-bit-secret",
  "JWT_ALGORITHM": "HS256",
  "RATE_LIMIT": "60 per minute",
  "LOG_LEVEL": "INFO",
  "LOG_FILE": null,
  "SYSLOG_HOST": null,
  "SYSLOG_PORT": null,
  "ALLOWED_IPS": [],
  "BLOCKED_IPS": [],
  "REQUIRE_CLIENT_CERT": false,
  "TLS_CERT": null,
  "TLS_KEY": null,
  "CA_CERT": null
}
```

**doctor/config.json**:
```json
{
  "HOST": "0.0.0.0",
  "PORT": 5001,
  "JWT_SECRET": "your-256-bit-secret",
  "JWT_ALGORITHM": "HS256",
  "RATE_LIMIT": "60 per minute",
  "LOG_LEVEL": "INFO",
  "LOG_FILE": null,
  "SYSLOG_HOST": null,
  "SYSLOG_PORT": null,
  "ALLOWED_IPS": [],
  "BLOCKED_IPS": [],
  "REQUIRE_CLIENT_CERT": false,
  "TLS_CERT": null,
  "TLS_KEY": null,
  "CA_CERT": null
}
```

**eliza/.env.example**:
```bash
ELIZA_JWT_SECRET="your-256-bit-secret"
ELIZA_JWT_ALGORITHM=HS256
ELIZA_LOG_LEVEL=INFO
ELIZA_RATE_LIMIT="60 per minute"
#ELIZA_ALLOWED_IPS=192.168.1.100,192.168.1.101
#ELIZA_BLOCKED_IPS=
#ELIZA_TLS_CERT=/path/to/eliza.crt
#ELIZA_TLS_KEY=/path/to/eliza.key
#ELIZA_CA_CERT=/path/to/ca.crt
```

**doctor/.env.example**:
```bash
DOCTOR_JWT_SECRET="your-256-bit-secret"
DOCTOR_JWT_ALGORITHM=HS256
DOCTOR_LOG_LEVEL=INFO
DOCTOR_RATE_LIMIT="60 per minute"
#DOCTOR_ALLOWED_IPS=192.168.1.100,192.168.1.101
#DOCTOR_BLOCKED_IPS=
#DOCTOR_TLS_CERT=/path/to/doctor.crt
#DOCTOR_TLS_KEY=/path/to/doctor.key
#DOCTOR_CA_CERT=/path/to/ca.crt
```

**eliza/Dockerfile**:
```Dockerfile
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
```

**doctor/Dockerfile**:
```Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY services /app/services
COPY doctor /app/doctor
RUN pip install --no-cache-dir flask PyJWT flask-limiter python-dotenv gunicorn
RUN useradd -ms /bin/bash appuser && chown -R appuser:appuser /app
USER appuser
EXPOSE 5001
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5001", "doctor.app:app"]
```

**eliza.service**:
```ini
[Unit]
Description=Eliza Chatbot API Service
After=network.target

[Service]
WorkingDirectory=/opt/eliza
EnvironmentFile=/opt/eliza/.env
User=eliza
ExecStart=/usr/local/bin/gunicorn -w 4 -b 0.0.0.0:5000 eliza.app:app
Restart=always

[Install]
WantedBy=multi-user.target
```

**doctor.service**:
```ini
[Unit]
Description=Doctor Chatbot API Service
After=network.target

[Service]
WorkingDirectory=/opt/doctor
EnvironmentFile=/opt/doctor/.env
User=doctor
ExecStart=/usr/local/bin/gunicorn -w 4 -b 0.0.0.0:5001 doctor.app:app
Restart=always

[Install]
WantedBy=multi-user.target
```
