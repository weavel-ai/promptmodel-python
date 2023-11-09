import os
from dotenv import load_dotenv

load_dotenv()
testmode: str = os.environ.get("TESTMODE", "false")
if testmode == "true":
    # ENDPOINT_URL = "https://7015-211-193-21-129.ngrok-free.app/api/cli"
    # ENDPOINT_URL = (
    #     "https://promptmodel-fastapi-server-weavel-be-pr-107.up.railway.app/api/cli"
    # )
    ENDPOINT_URL = "https://promptmodel.up.railway.app/api/cli"
    # WEB_CLIENT_URL = "http://localhost:3000"
    WEB_CLIENT_URL = "https://app.promptmodel.run"
else:
    ENDPOINT_URL = "https://promptmodel.up.railway.app/api/cli"
    WEB_CLIENT_URL = "https://app.promptmodel.run"

GRANT_ACCESS_URL = "https://app.promptmodel.run/cli/grant-access"
PROMPTMODEL_DEV_FILENAME = os.path.join(os.getcwd(), "promptmodel_dev.py")
PROMPTMODEL_DEV_STARTER_FILENAME = "STARTER.py"
