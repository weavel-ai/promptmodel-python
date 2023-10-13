import os
from dotenv import load_dotenv

load_dotenv()
deployment_stage: str = os.environ.get('DEPLOYMENT_STAGE')
if deployment_stage == 'local':
    ENDPOINT_URL = "https://privo.serveo.net/api/cli"
    GRANT_ACCESS_URL = "https://741a-121-140-205-66.ngrok-free.app/cli/grant-access"
    WEB_CLIENT_URL = "https://741a-121-140-205-66.ngrok-free.app"
else:
    ENDPOINT_URL = "https://promptmodel.up.railway.app/api/cli"
    GRANT_ACCESS_URL = "https://promptmodel.vercel.app/cli/grant-access"
    WEB_CLIENT_URL = "https://promptmodel.vercel.app"