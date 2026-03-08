import os
from dataclasses import dataclass, field


@dataclass
class Neo4jSettings:
    uri: str = field(default_factory=lambda: os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    user: str = field(default_factory=lambda: os.getenv("NEO4J_USER", "neo4j"))
    password: str = field(default_factory=lambda: os.getenv("NEO4J_PASSWORD", "cognitiveweave"))
    database: str = field(default_factory=lambda: os.getenv("NEO4J_DATABASE", "neo4j"))


@dataclass
class RedisSettings:
    host: str = field(default_factory=lambda: os.getenv("REDIS_HOST", "localhost"))
    port: int = field(default_factory=lambda: int(os.getenv("REDIS_PORT", "6379")))
    db: int = 0


@dataclass
class FAISSSettings:
    dimension: int = 384  # all-MiniLM-L6-v2 output dim
    index_path: str = field(default_factory=lambda: os.getenv("FAISS_INDEX_PATH", "data/faiss.index"))
    model_name: str = "all-MiniLM-L6-v2"


@dataclass
class OllamaSettings:
    base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_URL", "http://localhost:11434"))
    model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "llama3.2"))
    timeout: float = 30.0


@dataclass
class Settings:
    neo4j: Neo4jSettings = field(default_factory=Neo4jSettings)
    redis: RedisSettings = field(default_factory=RedisSettings)
    faiss: FAISSSettings = field(default_factory=FAISSSettings)
    ollama: OllamaSettings = field(default_factory=OllamaSettings)
