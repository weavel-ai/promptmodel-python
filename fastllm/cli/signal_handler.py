import signal
import sys
from fastllm.utils.config_utils import upsert_config

def dev_terminate_signal_handler(sig, frame):
    upsert_config({"online": False}, section="dev_branch")
    sys.exit(0)  