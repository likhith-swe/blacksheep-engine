"""
Black Sheep Ventures - Collective Intelligence Engine
Configuration Management
Loads configuration from .env file at project root.
"""

import os
from dotenv import load_dotenv

# Search for .env at project root
current_dir = os.path.dirname(os.path.abspath(__file__))
possible_env_paths = [
    os.path.join(current_dir, '../.env'),        # blacksheep-engine/.env
    os.path.join(current_dir, '../../.env'),     # parent/.env
    os.path.join(os.getcwd(), '.env'),           # cwd/.env
]

env_loaded = False
for env_path in possible_env_paths:
    if os.path.exists(env_path):
        load_dotenv(env_path, override=True)
        env_loaded = True
        break

if not env_loaded:
    load_dotenv(override=True)


class Config:
    """Core configuration for Black Sheep Ventures engine."""
    
    # Server & Security
    SECRET_KEY = os.environ.get('SECRET_KEY', 'blacksheep-ventures-secure-key-2026')
    DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    # Response encoding
    JSON_AS_ASCII = False
    
    # LLM Settings (OpenAI-compatible protocol)
    LLM_API_KEY = os.environ.get('LLM_API_KEY')
    LLM_BASE_URL = os.environ.get('LLM_BASE_URL', 'https://api.openai.com/v1')
    LLM_MODEL_NAME = os.environ.get('LLM_MODEL_NAME', 'gpt-4o-mini')
    
    # Optional High-Speed LLM for profile generation & lightweight tasks
    LLM_BOOST_API_KEY = os.environ.get('LLM_BOOST_API_KEY')
    LLM_BOOST_BASE_URL = os.environ.get('LLM_BOOST_BASE_URL')
    LLM_BOOST_MODEL_NAME = os.environ.get('LLM_BOOST_MODEL_NAME')
    
    # Zep Cloud Temporal Graph Memory
    ZEP_API_KEY = os.environ.get('ZEP_API_KEY')
    
    # Document Uploads & Storage
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    UPLOAD_FOLDER = os.path.join(BASE_DIR, '../uploads')
    ALLOWED_EXTENSIONS = {'pdf', 'md', 'txt', 'markdown'}
    
    # Text Chunking
    DEFAULT_CHUNK_SIZE = 500
    DEFAULT_CHUNK_OVERLAP = 50
    
    # Multi-Agent Social Simulation Settings (OASIS / CAMEL)
    OASIS_DEFAULT_MAX_ROUNDS = int(os.environ.get('OASIS_DEFAULT_MAX_ROUNDS', '10'))
    OASIS_SIMULATION_DATA_DIR = os.path.join(BASE_DIR, '../uploads/simulations')
    
    # Simulation Action Spaces
    OASIS_TWITTER_ACTIONS = [
        'CREATE_POST', 'LIKE_POST', 'REPOST', 'FOLLOW', 'DO_NOTHING', 'QUOTE_POST'
    ]
    OASIS_REDDIT_ACTIONS = [
        'LIKE_POST', 'DISLIKE_POST', 'CREATE_POST', 'CREATE_COMMENT',
        'LIKE_COMMENT', 'DISLIKE_COMMENT', 'SEARCH_POSTS', 'SEARCH_USER',
        'TREND', 'REFRESH', 'DO_NOTHING', 'FOLLOW', 'MUTE'
    ]
    
    # Autonomous ReACT Intelligence Report Settings
    REPORT_AGENT_MAX_TOOL_CALLS = int(os.environ.get('REPORT_AGENT_MAX_TOOL_CALLS', '5'))
    REPORT_AGENT_MAX_REFLECTION_ROUNDS = int(os.environ.get('REPORT_AGENT_MAX_REFLECTION_ROUNDS', '2'))
    REPORT_AGENT_TEMPERATURE = float(os.environ.get('REPORT_AGENT_TEMPERATURE', '0.5'))
    
    @classmethod
    def validate(cls) -> list[str]:
        """Validate required configuration settings."""
        errors: list[str] = []
        if not cls.LLM_API_KEY:
            errors.append("LLM_API_KEY is not configured in .env")
        if not cls.ZEP_API_KEY:
            errors.append("ZEP_API_KEY is not configured in .env")
        if os.environ.get("ZEP_API_URL"):
            errors.append("ZEP_API_URL is not supported; Black Sheep Engine integrates directly with Zep Cloud")
        if cls.DEBUG:
            import warnings
            warnings.warn("Flask DEBUG mode is enabled. Do not use in production.", RuntimeWarning)
        return errors
