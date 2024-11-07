from datetime import datetime
from typing import Optional, List, Dict
from pydantic import BaseModel, Field, validator
from config.constants import SummaryStatus, SummaryType

class SummaryKeyPoint(BaseModel):
    topic: str
    description: str
    timestamp: Optional[int]  # in seconds from start of meeting
    speaker: Optional[str]

class ActionItem(BaseModel):
    description: str
    assignee: Optional[str]
    due_date: Optional[datetime]
    status: str = "pending"
    priority: str = "medium"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    @validator('priority')
    def validate_priority(cls, v):
        allowed_priorities = ['low', 'medium', 'high', 'critical']
        if v.lower() not in allowed_priorities:
            raise ValueError(f'Priority must be one of: {", ".join(allowed_priorities)}')
        return v.lower()

class Decision(BaseModel):
    topic: str
    description: str
    made_by: Optional[str]
    timestamp: Optional[int]  # in seconds from start of meeting
    impact_areas: List[str] = Field(default_factory=list)
    stakeholders: List[str] = Field(default_factory=list)

class SummaryBase(BaseModel):
    meeting_id: int
    summary_type: SummaryType = SummaryType.FULL
    language: str = "en"
    key_points: List[SummaryKeyPoint] = Field(default_factory=list)
    action_items: List[ActionItem] = Field(default_factory=list)
    decisions: List[Decision] = Field(default_factory=list)
    full_text: str = Field(..., min_length=100)
    
    @validator('full_text')
    def validate_full_text(cls, v):
        if len(v.split()) < 50:  # Minimum 50 words
            raise ValueError('Summary text must contain at least 50 words')
        return v

class SummaryCreate(SummaryBase):
    generated_by: str = "ai-summit"
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    
    @validator('confidence_score')
    def validate_confidence(cls, v):
        if v < 0.0 or v > 1.0:
            raise ValueError('Confidence score must be between 0 and 1')
        return round(v, 2)

class SummaryUpdate(BaseModel):
    status: Optional[SummaryStatus]
    key_points: Optional[List[SummaryKeyPoint]]
    action_items: Optional[List[ActionItem]]
    decisions: Optional[List[Decision]]
    full_text: Optional[str]
    reviewed_by: Optional[str]
    review_notes: Optional[str]
    
    @validator('full_text')
    def validate_full_text_update(cls, v):
        if v is not None and len(v.split()) < 50:
            raise ValueError('Summary text must contain at least 50 words')
        return v

class SummaryInDB(SummaryBase):
    id: int
    status: SummaryStatus = SummaryStatus.PENDING
    generated_by: str
    confidence_score: float
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    reviewed_by: Optional[str]
    review_notes: Optional[str]
    review_date: Optional[datetime]
    metadata: Dict = Field(default_factory=dict)
    
    class Config:
        orm_mode = True

class SummaryResponse(SummaryInDB):
    word_count: int
    processing_time: Optional[float]  # in seconds
    
    @validator('word_count', always=True)
    def calculate_word_count(cls, v, values):
        if 'full_text' in values:
            return len(values['full_text'].split())
        return 0

class SummaryStats(BaseModel):
    total_summaries: int = 0
    average_length: int = 0  # in words
    average_confidence: float = 0.0
    average_processing_time: float = 0.0  # in seconds
    most_common_topics: List[str] = Field(default_factory=list)
    action_items_completion_rate: float = 0.0
    
    class Config:
        orm_mode = True

class SummaryFeedback(BaseModel):
    summary_id: int
    user_id: int
    rating: int = Field(..., ge=1, le=5)
    feedback_text: Optional[str] = Field(None, max_length=1000)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    @validator('rating')
    def validate_rating(cls, v):
        if v < 1 or v > 5:
            raise ValueError('Rating must be between 1 and 5')
        return v
    
    class Config:
        orm_mode = True

class SummaryTemplate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    template: str = Field(..., min_length=10)
    variables: List[str] = Field(default_factory=list)
    created_by: str
    is_default: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        orm_mode = True

class SummaryExport(BaseModel):
    summary_id: int
    export_format: str = "pdf"  # pdf, docx, html, txt
    include_metadata: bool = True
    include_feedback: bool = False
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    @validator('export_format')
    def validate_format(cls, v):
        allowed_formats = ['pdf', 'docx', 'html', 'txt']
        if v.lower() not in allowed_formats:
            raise ValueError(f'Export format must be one of: {", ".join(allowed_formats)}')
        return v.lower()
    
    class Config:
        orm_mode = True