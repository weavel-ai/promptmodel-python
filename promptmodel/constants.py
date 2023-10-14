import os
from dotenv import load_dotenv

load_dotenv()
deployment_stage: str = os.environ.get("DEPLOYMENT_STAGE")
if deployment_stage == "local":
    ENDPOINT_URL = "https://weavel.serveo.net/api/cli"
    WEB_CLIENT_URL = "http://localhost:3000"
else:
    ENDPOINT_URL = "https://promptmodel.up.railway.app/api/cli"
    WEB_CLIENT_URL = "https://promptmodel.vercel.app"
GRANT_ACCESS_URL = "https://promptmodel.vercel.app/cli/grant-access"
