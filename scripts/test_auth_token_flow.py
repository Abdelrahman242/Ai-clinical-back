from datetime import datetime, timezone

from jose import jwt

from app import auth
from app.config import JWT_ALGORITHM, JWT_SECRET_KEY

new_token = auth.create_access_token({"sub": "42"})
new_payload = jwt.decode(new_token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
assert new_payload["sub"] == "42"
assert new_payload["jti"]
assert "iat" in new_payload
assert datetime.fromtimestamp(new_payload["exp"], tz=timezone.utc) > datetime.now(timezone.utc)

legacy_token = auth.create_access_token({"sub": "old_username"})
legacy_payload = jwt.decode(legacy_token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
assert legacy_payload["sub"] == "old_username"

print("auth_token_flow=passed")
