from datetime import datetime
from typing import Optional, List, Dict
from pydantic import BaseModel, Field, validator
from config.constants import MeetingStatus, Platform, ContentType

class Participant(BaseModel):
    id: Optional[str]
    name: str
    email: Optional[str]
    role: Optional[str]
    join_time: Optional[datetime]
    leave_time: Optional[datetime]
    speaking_time: Optional[int] = Field(default=0)  # in seconds
    
    class Config:
        orm_mode = True

class MeetingContent(BaseModel):
    type: ContentType
    content_url: Optional[str]
    content_text: Optional[str]
    duration: Optional[int]  # in seconds
    language: Optional[str]
    processed: bool = False
    
    @validator('content_url', 'content_text')
    def validate_content(cls, v, values):
        if 'type' in values:
            if values['type'] in [ContentType.AUDIO, ContentType.TRANSCRIPT] and not (v or values.get('content_text')):
                raise ValueError(f'Either content_url or content_text must be provided for {values["type"]}')
        return v

class MeetingBase(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = Field(None, max_length=1000)
    platform: Platform
    platform_meeting_id: str
    scheduled_start: datetime
    scheduled_duration: int  # in minutes
    timezone: str = "UTC"
    organizer_id: str
    
    @validator('scheduled_duration')
    def validate_duration(cls, v):
        if v <= 0 or v > 1440:  # max 24 hours
            raise ValueError('Duration must be between 1 and 1440 minutes')
        return v

class MeetingCreate(MeetingBase):
    participants: Optional[List[Participant]]
    content: Optional[List[MeetingContent]]

class MeetingUpdate(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = Field(None, max_length=1000)
    scheduled_start: Optional[datetime]
    scheduled_duration: Optional[int]
    status: Optional[MeetingStatus]
    actual_start: Optional[datetime]
    actual_end: Optional[datetime]
    participants: Optional[List[Participant]]
    content: Optional[List[MeetingContent]]

class MeetingInDB(MeetingBase):
    id: int
    status: MeetingStatus = MeetingStatus.SCHEDULED
    actual_start: Optional[datetime]
    actual_end: Optional[datetime]
    participants: List[Participant]
    content: List[MeetingContent]
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict = Field(default_factory=dict)
    
    class Config:
        orm_mode = True

class MeetingResponse(MeetingInDB):
    duration: Optional[int]  # actual duration in minutes
    
    @validator('duration', always=True)
    def calculate_duration(cls, v, values):
        if values.get('actual_start') and values.get('actual_end'):
            return int((values['actual_end'] - values['actual_start']).total_seconds() / 60)
        return None

class MeetingStats(BaseModel):
    total_participants: int
    average_duration: float  # in minutes
    participant_speaking_times: Dict[str, int]  # participant_id: speaking_time_seconds
    content_types: List[ContentType]
    
    class Config:
        orm_mode = True

class MeetingFilter(BaseModel):
    start_date: Optional[datetime]
    end_date: Optional[datetime]
    status: Optional[MeetingStatus]
    platform: Optional[Platform]
    organizer_id: Optional[str]
    participant_email: Optional[str]
    
    class Config:
        orm_mode = True