from enum import Enum

class MeetingStatus(Enum):
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

class SummaryStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    APPROVED = "approved"
    REJECTED = "rejected"

class Platform(Enum):
    TEAMS = "teams"
    ZOOM = "zoom"

class Role(Enum):
    ADMIN = "admin"
    PRODUCT_OWNER = "product_owner"
    USER = "user"

class ContentType(Enum):
    AUDIO = "audio"
    TRANSCRIPT = "transcript"
    CHAT = "chat"

class SummaryType(Enum):
    FULL = "full"
    EXECUTIVE = "executive"
    ACTION_ITEMS = "action_items"
    DECISIONS = "decisions"

ERROR_MESSAGES = {
    "AUTH_FAILED": "Authentication failed",
    "INVALID_TOKEN": "Invalid or expired token",
    "PERMISSION_DENIED": "Permission denied",
    "RESOURCE_NOT_FOUND": "Resource not found",
    "INVALID_REQUEST": "Invalid request",
    "PROCESSING_ERROR": "Processing error occurred",
    "INTEGRATION_ERROR": "Integration error occurred",
}

SUCCESS_MESSAGES = {
    "SUMMARY_GENERATED": "Summary generated successfully",
    "SUMMARY_APPROVED": "Summary approved successfully",
    "MEETING_CREATED": "Meeting created successfully",
    "MEETING_UPDATED": "Meeting updated successfully",
}

SUMMARY_TEMPLATES = {
    "default": """
Meeting Summary
--------------
Date: {date}
Duration: {duration}
Participants: {participants}

Key Points:
{key_points}

Action Items:
{action_items}

Decisions Made:
{decisions}

Next Steps:
{next_steps}
    """.strip(),
    
    "executive": """
Executive Summary
----------------
Meeting: {title}
Date: {date}

Key Outcomes:
{outcomes}

Critical Decisions:
{decisions}

Required Actions:
{actions}
    """.strip(),
}

API_RATE_LIMITS = {
    "DEFAULT": "100/hour",
    "SUMMARY_GENERATION": "20/hour",
    "USER_OPERATIONS": "1000/day",
}

SUPPORTED_AUDIO_FORMATS = [
    "mp3",
    "wav",
    "m4a",
    "ogg"
]

SUPPORTED_LANGUAGES = [
    "en",  # English
    "es",  # Spanish
    "fr",  # French
    "de",  # German
    "it",  # Italian
    "pt",  # Portuguese
    "zh",  # Chinese
    "ja",  # Japanese
    "ko",  # Korean
]

MAX_RETRY_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 5

CACHE_TTL = {
    "summary": 3600,  # 1 hour
    "meeting": 1800,  # 30 minutes
    "user": 300,      # 5 minutes
}

VALIDATION_RULES = {
    "password_min_length": 8,
    "password_max_length": 128,
    "username_min_length": 3,
    "username_max_length": 50,
    "summary_min_length": 100,
    "summary_max_length": 5000,
}

DEFAULT_PAGINATION = {
    "page_size": 10,
    "max_page_size": 100,
}