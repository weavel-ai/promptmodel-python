import os
from dotenv import load_dotenv

load_dotenv()
testmode: str = os.environ.get("TESTMODE", "false")

if testmode == "true":
    ENDPOINT_URL = (
        os.environ.get(
            "TESTMODE_PROMPTMODEL_BACKEND_PUBLIC_URL", "http://localhost:8000"
        )
        + "/api/cli"
    )
    WEB_CLIENT_URL = os.environ.get(
        "TESTMODE_PROMPTMODEL_FRONTEND_PUBLIC_URL", "http://localhost:3000"
    )
    GRANT_ACCESS_URL = WEB_CLIENT_URL + "/cli/grant-access"
else:
    ENDPOINT_URL = (
        os.environ.get(
            "PROMPTMODEL_BACKEND_PUBLIC_URL", "https://promptmodel.up.railway.app"
        )
        + "/api/cli"
    )

    WEB_CLIENT_URL = os.environ.get(
        "PROMPTMODEL_FRONTEND_PUBLIC_URL", "https://app.promptmodel.run"
    )
    GRANT_ACCESS_URL = WEB_CLIENT_URL + "/cli/grant-access"

PROMPTMODEL_DEV_FILENAME = os.path.join(os.getcwd(), "promptmodel_dev.py")
PROMPTMODEL_DEV_STARTER_FILENAME = "STARTER.py"
