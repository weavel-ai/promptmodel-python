import signal
import sys
from promptmodel.utils.config_utils import upsert_config, read_config


def dev_terminate_signal_handler(sig, frame):
    config = read_config()
    print("\nTerminating...")
    if "dev_branch" in config:
        upsert_config({"online": False}, section="dev_branch")
        upsert_config({"initializing": False}, "dev_branch")
    sys.exit(0)
