import uuid
from datetime import datetime, timedelta, timezone

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from . import models
from .config import ACCESS_TOKEN_EXPIRE_MINUTES, JWT_ALGORITHM, JWT_SECRET_KEY
from .database import get_db

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def create_access_token(data: dict) -> str:
    """Create a JWT with immutable identity and a revocable token ID."""
    to_encode = data.copy()
    now = datetime.now(timezone.utc)
    expire = now + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.setdefault("jti", uuid.uuid4().hex)
    to_encode.update({"iat": now, "exp": expire})
    return jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def _credentials_exception() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="انتهت الجلسة أو بيانات الدخول غير صالحة. سجّل الدخول مرة أخرى.",
        headers={"WWW-Authenticate": "Bearer"},
    )


def get_current_user(
    token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)
) -> models.User:
    credentials_exception = _credentials_exception()
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        subject = payload.get("sub")
        if subject is None:
            raise credentials_exception
    except JWTError as exc:
        raise credentials_exception from exc

    jti = payload.get("jti")
    if jti and db.query(models.RevokedToken).filter(models.RevokedToken.jti == jti).first():
        raise credentials_exception

    # New tokens use the immutable numeric user ID. The username fallback is
    # intentionally kept for tokens issued before this migration.
    try:
        user_id = int(subject)
    except (TypeError, ValueError):
        user_id = None

    if user_id is not None:
        user = db.query(models.User).filter(models.User.id == user_id).first()
    else:
        user = db.query(models.User).filter(models.User.username == subject).first()

    if user is None or not user.is_active:
        raise credentials_exception
    return user


def revoke_token(token: str, user: models.User, db: Session) -> None:
    """Persist logout for tokens issued by the current auth implementation."""
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
    except JWTError:
        return

    jti = payload.get("jti")
    exp = payload.get("exp")
    if not jti or not exp:
        # Legacy tokens have no jti; the client still deletes them locally.
        return

    exists = db.query(models.RevokedToken).filter(models.RevokedToken.jti == jti).first()
    if exists is None:
        db.add(
            models.RevokedToken(
                jti=jti,
                user_id=user.id,
                expires_at=datetime.fromtimestamp(exp, tz=timezone.utc),
            )
        )
        db.commit()


def require_admin(current_user: models.User = Depends(get_current_user)) -> models.User:
    """Only active administrators can manage system-owned source documents."""
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="العملية دي متاحة للأدمن بس (إدارة مستندات النظام)",
        )
    return current_user
