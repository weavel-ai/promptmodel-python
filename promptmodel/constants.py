import os
from dotenv import load_dotenv

load_dotenv()
testmode: str = os.environ.get("TESTMODE", "false")
if testmode == "true":
    # ENDPOINT_URL = "https://2451-124-49-14-184.ngrok-free.app/api/cli"
    ENDPOINT_URL = "https://weavel.serveo.net/api/cli"
    # ENDPOINT_URL = "https://promptmodel-colab.up.railway.app/api/cli"
    # ENDPOINT_URL = "http://localhost:8443/api/cli"

    # WEB_CLIENT_URL = "http://localhost:3000"
    WEB_CLIENT_URL = "https://promptmodel.serveo.net"
    # WEB_CLIENT_URL = "https://app.promptmodel.run"
    # GRANT_ACCESS_URL = "http://localhost:3000/cli/grant-access"
    GRANT_ACCESS_URL = "https://promptmodel.serveo.net/cli/grant-access"
    # GRANT_ACCESS_URL = "https://app.promptmodel.run/cli/grant-access"
else:
    ENDPOINT_URL = "https://promptmodel.up.railway.app/api/cli"
    WEB_CLIENT_URL = "https://app.promptmodel.run"

    GRANT_ACCESS_URL = "https://app.promptmodel.run/cli/grant-access"

PROMPTMODEL_DEV_FILENAME = os.path.join(os.getcwd(), "promptmodel_dev.py")
PROMPTMODEL_DEV_STARTER_FILENAME = "STARTER.py"
