from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, EmailStr, Field, validator
from config.constants import Role

class UserBase(BaseModel):
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50)
    full_name: str = Field(..., min_length=1, max_length=100)
    role: Role = Field(default=Role.USER)
    organization: Optional[str] = Field(default=None, max_length=100)
    
    @validator('username')
    def username_alphanumeric(cls, v):
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError('Username must be alphanumeric')
        return v

class UserCreate(UserBase):
    password: str = Field(..., min_length=8, max_length=128)
    
    @validator('password')
    def password_strength(cls, v):
        if not any(char.isupper() for char in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(char.islower() for char in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(char.isdigit() for char in v):
            raise ValueError('Password must contain at least one number')
        return v

class UserUpdate(BaseModel):
    full_name: Optional[str] = Field(None, min_length=1, max_length=100)
    organization: Optional[str] = Field(None, max_length=100)
    role: Optional[Role] = None
    is_active: Optional[bool] = None

class UserInDB(UserBase):
    id: int
    hashed_password: str
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None
    
    class Config:
        orm_mode = True

class UserResponse(UserBase):
    id: int
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime]
    
    class Config:
        orm_mode = True

class UserPreferences(BaseModel):
    language: str = Field(default="en")
    timezone: str = Field(default="UTC")
    notification_enabled: bool = Field(default=True)
    summary_format: str = Field(default="default")
    auto_approve_summaries: bool = Field(default=False)
    
    class Config:
        orm_mode = True

class UserSession(BaseModel):
    user_id: int
    session_id: str
    token: str
    expires_at: datetime
    ip_address: Optional[str]
    user_agent: Optional[str]
    
    class Config:
        orm_mode = True

class UserStats(BaseModel):
    total_meetings: int = 0
    total_summaries_generated: int = 0
    total_summaries_approved: int = 0
    average_summary_length: float = 0
    most_active_platform: Optional[str] = None
    
    class Config:
        orm_mode = True