import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Security settings
SECRET_KEY = os.getenv('AI_SUMMIT_SECRET_KEY', 'your-secret-key-here')
DEBUG = os.getenv('AI_SUMMIT_DEBUG', 'False').lower() == 'true'

# API settings
API_VERSION = 'v1'
API_PREFIX = f'/api/{API_VERSION}'

# Database settings
DATABASE_URL = os.getenv('AI_SUMMIT_DATABASE_URL', 'sqlite:///./ai_summit.db')

# AI Model settings
LLM_MODEL = os.getenv('AI_SUMMIT_LLM_MODEL', 'gpt-4')
SPEECH_TO_TEXT_MODEL = os.getenv('AI_SUMMIT_STT_MODEL', 'whisper-1')
MAX_SUMMARY_LENGTH = int(os.getenv('AI_SUMMIT_MAX_SUMMARY_LENGTH', '2000'))
MIN_SUMMARY_LENGTH = int(os.getenv('AI_SUMMIT_MIN_SUMMARY_LENGTH', '200'))

# Integration settings
TEAMS_CLIENT_ID = os.getenv('AI_SUMMIT_TEAMS_CLIENT_ID', '')
TEAMS_CLIENT_SECRET = os.getenv('AI_SUMMIT_TEAMS_CLIENT_SECRET', '')
TEAMS_TENANT_ID = os.getenv('AI_SUMMIT_TEAMS_TENANT_ID', '')

ZOOM_CLIENT_ID = os.getenv('AI_SUMMIT_ZOOM_CLIENT_ID', '')
ZOOM_CLIENT_SECRET = os.getenv('AI_SUMMIT_ZOOM_CLIENT_SECRET', '')
ZOOM_ACCOUNT_ID = os.getenv('AI_SUMMIT_ZOOM_ACCOUNT_ID', '')

# Storage settings
STORAGE_PATH = os.getenv('AI_SUMMIT_STORAGE_PATH', str(BASE_DIR / 'storage'))
TEMP_PATH = os.getenv('AI_SUMMIT_TEMP_PATH', str(BASE_DIR / 'temp'))

# Logging settings
LOG_LEVEL = os.getenv('AI_SUMMIT_LOG_LEVEL', 'INFO')
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE = os.getenv('AI_SUMMIT_LOG_FILE', str(BASE_DIR / 'logs' / 'ai_summit.log'))

# Performance settings
WORKER_COUNT = int(os.getenv('AI_SUMMIT_WORKER_COUNT', '4'))
BATCH_SIZE = int(os.getenv('AI_SUMMIT_BATCH_SIZE', '10'))
REQUEST_TIMEOUT = int(os.getenv('AI_SUMMIT_REQUEST_TIMEOUT', '30'))

# Security settings
ALLOWED_HOSTS = os.getenv('AI_SUMMIT_ALLOWED_HOSTS', '*').split(',')
CORS_ORIGINS = os.getenv('AI_SUMMIT_CORS_ORIGINS', '*').split(',')
TOKEN_EXPIRY = int(os.getenv('AI_SUMMIT_TOKEN_EXPIRY', '3600'))  # 1 hour