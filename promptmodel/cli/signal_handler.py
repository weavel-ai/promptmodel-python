import signal
import sys
from promptmodel.utils.config_utils import upsert_config, read_config


def dev_terminate_signal_handler(sig, frame):
    config = read_config()
    print("\nTerminating...")
    if "connection" in config:
        upsert_config({"online": False}, section="connection")
        upsert_config({"initializing": False}, "connection")
    sys.exit(0)
