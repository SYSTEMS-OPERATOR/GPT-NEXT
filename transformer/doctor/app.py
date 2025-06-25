"""Doctor service exposing a REST API backed by an ELIZA-style chatbot."""

try:
    from flask import Flask, request, jsonify, g
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address
except ModuleNotFoundError as exc:
    raise SystemExit("Flask is required to run this service") from exc
import jwt
import sys
import ssl
import uuid
from werkzeug.exceptions import HTTPException

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
    try:
        bot_reply = generate_response(user_msg, persona="doctor")
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
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
