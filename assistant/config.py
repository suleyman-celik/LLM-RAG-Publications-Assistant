from __future__ import annotations

import os
import inspect
from dataclasses import dataclass, fields, field, asdict
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import logging
# ---------------- Logging ----------------
logging.basicConfig(
    level=logging.INFO,  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv()


@dataclass(frozen=True)
class Settings:
    """
    Examples
    --------
    s = Settings()

    print(s.as_list)
    # ['llm_rag', 'localhost', ...]

    print(s.as_dict)
    # {'POSTGRES_DB': 'llm_rag', 'POSTGRES_HOST': 'localhost', ...}

    print(Settings.field_names())
    # ['POSTGRES_DB', 'POSTGRES_HOST', ...]

    print(Settings.defaults_as_dict())
    # {'POSTGRES_DB': 'llm_rag', 'POSTGRES_HOST': 'localhost', ...}
    """
    def __repr__(self) -> str:
        """Safe string representation with masked secrets."""
        values = []
        for f in fields(self):
            val = getattr(self, f.name)
            if f.metadata.get("secret"):
                val = "****"
            values.append(f"{f.name}={val!r}")
        return f"{self.__class__.__name__}({', '.join(values)})"

    @classmethod
    def field_names(cls) -> list:
        """Return field names as a list (class-level)."""
        return [f.name for f in fields(cls)]
    
    @property
    def as_list(self) -> list:
        """Return all values as a list (mask secrets)."""
        return [
            "****" if f.metadata.get("secret") else getattr(self, f.name)
            for f in fields(self)
        ]

    @property
    def as_dict(self) -> dict:
        """Return all values as a dict (mask secrets)."""
        # return asdict(self)
        return {
            f.name: ("****" if f.metadata.get("secret") else getattr(self, f.name))
            for f in fields(self)
        }

    @classmethod
    def defaults_as_list(cls) -> list:
        """Return default field values as a list (class-level) (mask secrets)."""
        return [
            "****" if f.metadata.get("secret") else f.default
            for f in fields(cls)
        ]

    @classmethod
    def defaults_as_dict(cls) -> dict:
        """Return default field values as a dict (class-level) (mask secrets)."""
        return {
            f.name: ("****" if f.metadata.get("secret") else f.default)
            for f in fields(cls)
        }

    def local_or_docker_service(url="", service=None):
        if os.path.exists('/.dockerenv') and service:
            return url.replace("localhost", service)
        elif service:
            return url.replace(service, "localhost")
        else:
            return url
    
    # def resolved_provider(self) -> str:
    #     # Auto-fallback to Ollama if OpenAI key is missing or provider explicitly set to "ollama"
    #     if self.OPENAI_API_KEY and self.LLM_PROVIDER != "ollama":
    #         return "openai"
    #     return "ollama"
    
    ## re-declare getter func_name.getter → redefine the getter of an existing property func_name.
    # @property
    # def OPENAI_API_KEY(self) -> str:
    #     """Public property → always masked"""
    #     return "****"
    # @property
    # def OPENAI_API_KEY(self, internal: bool = False) -> str:
    #     """Internal use only, property → real secret (use inside lib only) (not part of public API)."""
    #     # caller = inspect.stack()[1].filename
    #     # if "yourlib/" not in caller:  # adjust path/package name
    #     # if not internal:
    #     #     raise PermissionError("Access to secret_key is restricted for external use.")
    #     return self.__OPENAI_API_KEY

    DATA_URL: str = os.getenv("DATA_URL", "https://www.who.int/europe/publications/i")

    TZ_INFO: str = os.getenv("TZ", "Europe/Istanbul")
    TZ_LOCAL: datetime = datetime.now(ZoneInfo(str(TZ_INFO)))  # pytz.timezone("Europe/Istanbul")
    TZ_UTC: datetime = datetime.now(timezone.utc)
        
    # Postgres Configuration
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "llm_rag")
    POSTGRES_HOST: str = local_or_docker_service(os.getenv("POSTGRES_HOST", "localhost"), "postgres")  # "localhost" or "postgres"
    POSTGRES_PORT: int = int(os.getenv("POSTGRES_PORT", "5432"))
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "admin")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "admin")
    POSTGRES_TABLE: str = os.getenv("POSTGRES_TABLE", "text_chunks")

    # Grafana Configuration
    GRAFANA_ADMIN_USER: str = os.getenv("GRAFANA_ADMIN_USER", "admin")
    GRAFANA_ADMIN_PASSWORD: str = os.getenv("GRAFANA_ADMIN_PASSWORD", "admin")
    GRAFANA_SECRET_KEY: str = os.getenv("GRAFANA_SECRET_KEY", "SECRET_KEY")

    # OpenAI Configuration
    # Actually hide attribute with __double_underscore (name mangling), _single_underscore signals "protected"
    OPENAI_API_KEY: str | None = field(default=os.getenv("OPENAI_API_KEY", "sk-..."), metadata={"secret": True}, repr=False)
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    OPENAI_MODEL_EMBED: str = os.getenv("OPENAI_MODEL_EMBED", "text-embedding-3-small")
    OPENAI_MODEL_CHAT: str = os.getenv("OPENAI_MODEL_CHAT", "gpt-4o-mini")

    # huggingface Configuration
    HF_TOKEN: str | None = field(default=os.getenv("HF_TOKEN", "sk-..."), metadata={"secret": True}, repr=False)
    HF_API_KEY: str | None = field(default=os.getenv("HF_TOKEN", "sk-..."), metadata={"secret": True}, repr=False)  # alias
    HF_BASE_URL: str = os.getenv("HF_BASE_URL", "https://router.huggingface.co/v1")  # for openai
    HF_MODEL_EMBED: str = os.getenv("HF_MODEL_EMBED", "nomic-ai/nomic-embed-text-v1.5")
    HF_MODEL_CHAT: str = os.getenv("HF_MODEL_CHAT", "openai/gpt-oss-120b")

    # Ollama Configuration
    OLLAMA_API_KEY: str | None = os.getenv("OLLAMA_API_KEY", "ollama")  # dummy key
    OLLAMA_BASE_URL: str = local_or_docker_service(os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"), "ollama")  # "localhost" or "ollama"
    OLLAMA_MODEL_EMBED: str = os.getenv("OLLAMA_MODEL_EMBED", "nomic-embed-text")
    OLLAMA_MODEL_CHAT: str = os.getenv("OLLAMA_MODEL_CHAT", "phi3")

    # LLM RAG MODEL Configuration
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "OLLAMA")  #.lower()  ## OPENAI or OLLAMA or HF
    API_KEY: str | None = field(default=os.getenv(f"{LLM_PROVIDER}_API_KEY", OLLAMA_API_KEY), metadata={"secret": True}, repr=False)
    BASE_URL: str = local_or_docker_service(os.getenv(f"{LLM_PROVIDER}_BASE_URL", OLLAMA_BASE_URL), "ollama")
    MODEL_EMBED: str = os.getenv(f"{LLM_PROVIDER}_MODEL_EMBED", OLLAMA_MODEL_EMBED)
    MODEL_CHAT: str = os.getenv(f"{LLM_PROVIDER}_MODEL_CHAT", OLLAMA_MODEL_CHAT)

        
    # Scraper
    TIMEOUT_PDF_REQUEST: int = int(os.getenv("TIMEOUT_PDF_REQUEST", 60))
    SE_SLOWMO_MS: int = int(os.getenv("SE_SLOWMO_MS", 200))
    DEFAULT_MAX_PAGES: int = int(os.getenv("DEFAULT_MAX_PAGES", 1))

    # Chunking
    CHUNK_MAX_TOKENS: int = int(os.getenv("CHUNK_MAX_TOKENS", 512))
    CHUNK_OVERLAP_TOKENS: int = int(os.getenv("CHUNK_OVERLAP_TOKENS", 64))

    # Features
    ENABLE_FAISS: bool = os.getenv("ENABLE_FAISS", "false").lower() == "true"


SETTINGS = Settings()  # instance

## Database Config
DB_CONFIG = {
    "database": Settings().POSTGRES_DB,  # "dbname"
    "host": Settings().POSTGRES_HOST,
    "port": Settings().POSTGRES_PORT,
    "user": Settings().POSTGRES_USER,
    "password": Settings().POSTGRES_PASSWORD,
}