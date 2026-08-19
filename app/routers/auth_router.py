from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from .. import auth, models, schemas
from ..database import get_db

router = APIRouter(prefix="/api/v1/auth", tags=["Auth"])


@router.post("/register", response_model=schemas.UserResponse)
def register(user: schemas.UserCreate, db: Session = Depends(get_db)):
    existing = db.query(models.User).filter(models.User.username == user.username).first()
    if existing:
        raise HTTPException(status_code=400, detail="اسم المستخدم موجود بالفعل")

    # أول يوزر يتسجل في السيستم بيبقى أدمن تلقائيًا (bootstrap) — أي حد بعد كده user عادي
    is_first_user = db.query(models.User).count() == 0

    new_user = models.User(
        username=user.username,
        email=user.email,
        hashed_password=auth.hash_password(user.password),
        is_admin=is_first_user,
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user


@router.post("/login", response_model=schemas.Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.username == form_data.username).first()
    if not user or not auth.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="اسم المستخدم أو الباسورد غلط")

    token = auth.create_access_token(data={"sub": str(user.id)})
    return schemas.Token(access_token=token)


@router.get("/me", response_model=schemas.UserResponse)
def get_me(current_user: models.User = Depends(auth.get_current_user)):
    return current_user


@router.post("/logout")
def logout(
    token: str = Depends(auth.oauth2_scheme),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user),
):
    """Revoke the current JWT and let the client remove its local copy."""
    auth.revoke_token(token, current_user, db)
    return {"status": "logged_out"}


@router.put("/me", response_model=schemas.UserResponse)
@router.patch("/me", response_model=schemas.UserResponse)
def update_me(
    update: schemas.UserUpdate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user),
):
    requested_username = update.username or update.name
    if requested_username:
        existing = (
            db.query(models.User)
            .filter(
                models.User.username == requested_username,
                models.User.id != current_user.id,
            )
            .first()
        )
        if existing:
            raise HTTPException(status_code=400, detail="اسم المستخدم ده مستخدم بالفعل")
        current_user.username = requested_username

    if update.password:
        current_user.hashed_password = auth.hash_password(update.password)

    db.commit()
    db.refresh(current_user)

    # Return a fresh token so clients can replace any legacy username-based
    # token immediately after a profile change.
    current_user.access_token = auth.create_access_token(data={"sub": str(current_user.id)})
    return current_user
