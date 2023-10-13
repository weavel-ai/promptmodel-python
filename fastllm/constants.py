import os
from dotenv import load_dotenv

load_dotenv()
deployment_stage: str = os.environ.get('DEPLOYMENT_STAGE')
if deployment_stage == 'local':
	ENDPOINT_URL = "https://privo.serveo.net/api/cli"
else:
	ENDPOINT_URL = "https://fastllm.up.railway.app/api/cli"
GRANT_ACCESS_URL = "https://741a-121-140-205-66.ngrok-free.app/cli/grant-access"
WEB_CLIENT_URL = "https://741a-121-140-205-66.ngrok-free.app"