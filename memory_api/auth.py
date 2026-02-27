from fastapi import HTTPException, Request, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from memory_api.config import settings

_bearer = HTTPBearer(auto_error=False)


async def require_auth(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Security(_bearer),
) -> None:
    if not settings.api_auth_token:
        return
    if credentials is None or credentials.credentials != settings.api_auth_token:
        raise HTTPException(status_code=401, detail="Unauthorized")
