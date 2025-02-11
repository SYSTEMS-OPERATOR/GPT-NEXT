# ELIZA & DOCTOR Chatbot Services

## Overview

This repository contains two chatbot API services, **ELIZA** and **DOCTOR**, inspired by the classic ELIZA psychotherapy chatbot. Each service runs independently and provides conversational responses in the style of a Rogerian psychotherapist. Both services share common code for configuration, logging, and conversation logic, but they are isolated into separate deployments (microservices) for flexibility and modularity.

**Key Features:**

- **Stateless REST API:** Each service exposes an HTTP API endpoint for chatbot interaction. The services are stateless, with no session stored on the server, relying on JWT tokens for any user context.
- **Configuration via Environment**: All secrets and configurable parameters are provided through environment variables (with a `.env` file for local development) following the Twelve-Factor App principles&#8203;:contentReference[oaicite:0]{index=0}. A JSON config file is available as a secondary fallback for convenience.
- **Secure Authentication:** The APIs use JSON Web Tokens (JWT) for authentication of requests. Only requests with a valid JWT (in the `Authorization: Bearer ...` header) are processed, ensuring that only authorized clients can interact.
- **Rate Limiting:** To prevent abuse and denial-of-service, the services enforce configurable rate limiting on the chat endpoint. This helps maintain availability and fairness, and protects against brute-force attacks&#8203;:contentReference[oaicite:1]{index=1}.
- **Mutual TLS Support:** For environments requiring high security, the services can be configured to require mutual TLS authentication, ensuring both client and server validate each other’s certificates&#8203;:contentReference[oaicite:2]{index=2}.
- **IP Whitelisting/Blacklisting:** The services can restrict access based on client IP addresses. Only allowed IPs can access the API (or conversely, blocked IPs can be denied), providing an extra layer of security&#8203;:contentReference[oaicite:3]{index=3}.
- **Structured Logging:** All services emit logs in structured JSON format, including timestamps and unique request IDs for each request. These **forensic logs** include correlation identifiers and key request metadata, aiding in audit and forensic analysis&#8203;:contentReference[oaicite:4]{index=4}. Logs can be sent to local files or forwarded to centralized systems (e.g., ELK stack) for monitoring.
- **Container & Daemon Ready:** Each service comes with a Dockerfile for containerization and a systemd unit file for running as a Linux service. This offers flexibility in deployment, whether you prefer Docker, Kubernetes, or traditional VM/metal deployments with systemd.

## Repository Structure

