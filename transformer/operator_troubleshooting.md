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

- **Regular Patching:** Keep the system and application dependencies updated. This includes applying OS updates (for the base Docker image or host OS) and updating Python packages for any security fixes&#8203;:contentReference[oaicite:16]{index=16}.
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

