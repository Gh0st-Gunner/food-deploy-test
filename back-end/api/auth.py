import hashlib
import os
import secrets
from datetime import datetime, timedelta
from fastapi import Header, HTTPException, Depends, status
from sqlalchemy.orm import Session

from core.database import get_session, User, UserSession


def hash_password(password: str) -> str:
    """Hash password using PBKDF2-HMAC-SHA256 with random salt."""
    salt = os.urandom(16)
    key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100000)
    return salt.hex() + ":" + key.hex()


def verify_password(password: str, hashed: str) -> bool:
    """Verify standard PBKDF2 hashed password."""
    try:
        salt_hex, key_hex = hashed.split(":")
        salt = bytes.fromhex(salt_hex)
        key = bytes.fromhex(key_hex)
        new_key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100000)
        return new_key == key
    except Exception:
        return False


def create_session(user_id: str) -> str:
    """Create a new database-backed session token valid for 7 days."""
    token = secrets.token_hex(32)
    db = get_session()
    try:
        # Clear existing sessions for this user to keep database clean (optional, but good practice)
        db.query(UserSession).filter(UserSession.user_id == user_id).delete()
        
        session = UserSession(
            session_token=token,
            user_id=user_id,
            expires_at=datetime.utcnow() + timedelta(days=7),
        )
        db.add(session)
        db.commit()
        return token
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def destroy_session(token: str):
    """Delete session token on logout."""
    db = get_session()
    try:
        db.query(UserSession).filter(UserSession.session_token == token).delete()
        db.commit()
    except Exception:
        db.rollback()
    finally:
        db.close()


async def get_current_user(authorization: str = Header(None)) -> User:
    """FastAPI dependency to extract and validate session token from Authorization header."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid authentication token. Header format: 'Authorization: Bearer <token>'",
        )
    
    token = authorization.split(" ")[1]
    db = get_session()
    try:
        session = db.query(UserSession).filter(UserSession.session_token == token).first()
        if not session:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Session not found or has logged out",
            )
            
        if session.expires_at < datetime.utcnow():
            db.delete(session)
            db.commit()
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Session has expired. Please log in again",
            )
            
        user = db.query(User).filter(User.id == session.user_id).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User associated with session not found",
            )
            
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User account is deactivated/blocked",
            )
            
        # Return detached user object or copy of it so it remains accessible outside session context
        return User(
            id=user.id,
            username=user.username,
            role=user.role,
            is_active=user.is_active,
            created_at=user.created_at,
        )
    finally:
        db.close()


async def get_current_admin(current_user: User = Depends(get_current_user)) -> User:
    """FastAPI dependency to verify user has administrative role."""
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrative privileges required to perform this action",
        )
    return current_user
