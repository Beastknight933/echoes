"""
Configuration module using Pydantic Settings for type-safe environment variables.
Updated with OpenRouter support.
"""
from pydantic_settings import BaseSettings
from pydantic import Field, validator
from typing import List, Optional
from pathlib import Path
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Server Configuration
    port: int = Field(default=8000, env="PORT")
    host: str = Field(default="0.0.0.0", env="HOST")
    reload: bool = Field(default=True, env="RELOAD")
    
    # CORS Configuration
    allowed_origins: str = Field(default="*", env="ALLOWED_ORIGINS")
    
    @property
    def cors_origins(self) -> List[str]:
        """Parse allowed origins into a list."""
        if self.allowed_origins == "*":
            return ["*"]
        return [origin.strip() for origin in self.allowed_origins.split(",")]
    
    # OpenRouter Configuration (PRIMARY)
    openrouter_api_key: Optional[str] = Field(default=None, env="OPENROUTER_API_KEY")
    openrouter_model: str = Field(
        default="deepseek/deepseek-chat",  # DeepSeek V3.1 (free & excellent)
        env="OPENROUTER_MODEL"
    )
    openrouter_site_url: Optional[str] = Field(default=None, env="OPENROUTER_SITE_URL")
    openrouter_app_name: str = Field(default="Echoes-Backend", env="OPENROUTER_APP_NAME")
    
    # Alternative LLM Providers (fallback)
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", env="OPENAI_MODEL")
    
    deepseek_api_key: Optional[str] = Field(default=None, env="DEEPSEEK_API_KEY")
    deepseek_model: str = Field(default="deepseek-chat", env="DEEPSEEK_MODEL")
    
    gemini_api_key: Optional[str] = Field(default=None, env="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-1.5-flash", env="GEMINI_MODEL")
    
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    anthropic_model: str = Field(default="claude-3-5-sonnet-20241022", env="ANTHROPIC_MODEL")
    
    # External API Configuration
    wiktionary_api_base: str = Field(
        default="https://en.wiktionary.org/api/rest_v1",
        env="WIKTIONARY_API_BASE"
    )
    etymonline_base_url: str = Field(
        default="https://www.etymonline.com",
        env="ETYMONLINE_BASE_URL"
    )
    
    # Model Configuration
    sentence_transformer_model: str = Field(
        default="paraphrase-MiniLM-L6-v2",
        env="SENTENCE_TRANSFORMER_MODEL"
    )
    
    # Data Paths
    embeddings_dir: str = Field(default="embeddings", env="EMBEDDINGS_DIR")
    data_dir: str = Field(default="data", env="DATA_DIR")
    assets_dir: str = Field(default="assets", env="ASSETS_DIR")
    
    # Cache Configuration
    use_redis: bool = Field(default=False, env="USE_REDIS")
    redis_host: str = Field(default="localhost", env="REDIS_HOST")
    redis_port: int = Field(default=6379, env="REDIS_PORT")
    redis_db: int = Field(default=0, env="REDIS_DB")
    redis_ttl: int = Field(default=3600, env="REDIS_TTL")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default="logs/echoes.log", env="LOG_FILE")
    
    # Rate Limiting
    rate_limit_enabled: bool = Field(default=True, env="RATE_LIMIT_ENABLED")
    rate_limit_per_minute: int = Field(default=60, env="RATE_LIMIT_PER_MINUTE")
    
    # Feature Flags
    use_llm_etymology: bool = Field(default=True, env="USE_LLM_ETYMOLOGY")
    use_external_apis: bool = Field(default=True, env="USE_EXTERNAL_APIS")
    fallback_to_csv: bool = Field(default=True, env="FALLBACK_TO_CSV")
    
    # Constants
    DEFAULT_TOP_N: int = 6
    MAX_TOP_N: int = 50
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    @property
    def active_llm_provider(self) -> Optional[str]:
        """Determine which LLM provider is configured (OpenRouter takes priority)."""
        if self.openrouter_api_key:
            return "openrouter"
        elif self.openai_api_key:
            return "openai"
        elif self.deepseek_api_key:
            return "deepseek"
        elif self.gemini_api_key:
            return "gemini"
        elif self.anthropic_api_key:
            return "anthropic"
        return None
    
    @property
    def embeddings_path(self) -> Path:
        return Path(self.embeddings_dir)
    
    @property
    def data_path(self) -> Path:
        return Path(self.data_dir)
    
    @property
    def assets_path(self) -> Path:
        return Path(self.assets_dir)
    
    def ensure_directories(self):
        """Create necessary directories if they don't exist."""
        self.embeddings_path.mkdir(parents=True, exist_ok=True)
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.assets_path.mkdir(parents=True, exist_ok=True)
        
        # Create logs directory
        log_dir = Path(self.log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()

# Ensure directories exist on import
settings.ensure_directories()