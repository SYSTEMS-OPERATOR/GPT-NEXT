"""JSON logging helpers for the microservices."""

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
