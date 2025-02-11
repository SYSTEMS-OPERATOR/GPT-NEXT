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
