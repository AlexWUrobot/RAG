"""
SensorDoc-AI: RAG Pipeline for Hardware Datasheets

Parses sensor datasheets (PDF), preserves table structures, stores embeddings
in ChromaDB, and provides hybrid retrieval (semantic + BM25) for accurate
technical term lookup.
"""

import os
import re
import sys
from urllib import error as urllib_error
from urllib import request as urllib_request
from collections import Counter

os.environ.setdefault("PYMUPDF_SUGGEST_LAYOUT_ANALYZER", "0")

import fitz  # PyMuPDF
import yaml
from chromadb.config import Settings
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain.retrievers import EnsembleRetriever
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from reasoning_pipeline import ReasoningPipeline

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()
with open(os.path.join(os.path.dirname(__file__), "config.yaml"), "r") as f:
    config = yaml.safe_load(f)

rag_cfg = config["rag_module"]

# Keep PyMuPDF from printing advisory messages during extraction.
fitz.TOOLS.mupdf_display_warnings(False)
fitz.TOOLS.mupdf_display_errors(False)


class PromptInjectionGuard:
    """Small guardrail layer for suspicious queries and untrusted context."""

    _query_patterns = [
        re.compile(pattern, re.IGNORECASE)
        for pattern in [
            r"ignore .*instructions",
            r"(reveal|show|print|dump).*(api[ _-]?key|token|secret|password)",
            r"(tell|talk|explain|describe).*(api[ _-]?key|token|secret|password)",
            r"(api[ _-]?key|token|secret|password).*(tell|talk|explain|describe|show|reveal|print|dump)",
            r"your (api[ _-]?key|token|secret|password)",
            r"(what|where).*(api[ _-]?key|token|secret|password)",
            r"(system|developer|hidden) prompt",
            r"openai_api_key|os\.environ|process\.env|authorization",
            r"bypass|override|jailbreak",
        ]
    ]
    _context_patterns = [
        re.compile(pattern, re.IGNORECASE)
        for pattern in [
            r"ignore .*instructions",
            r"system prompt",
            r"developer message",
            r"reveal .*secret",
            r"api[ _-]?key",
        ]
    ]
    _secret_patterns = [
        re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
        re.compile(r"OPENAI_API_KEY", re.IGNORECASE),
        re.compile(r"(api[ _-]?key|token|secret|password)", re.IGNORECASE),
    ]

    @classmethod
    def validate_question(cls, question: str) -> str | None:
        if not rag_cfg.get("security", {}).get("block_suspicious_queries", True):
            return None
        for pattern in cls._query_patterns:
            if pattern.search(question):
                return (
                    "Request blocked by the prompt-injection guard. "
                    "Ask a question about the datasheet content only."
                )
        return None

    @classmethod
    def sanitize_context(cls, context: str) -> str:
        if not rag_cfg.get("security", {}).get("prompt_injection_guard", True):
            return context

        safe_lines: list[str] = []
        for line in context.splitlines():
            if any(pattern.search(line) for pattern in cls._context_patterns):
                continue
            safe_lines.append(line)
        return "\n".join(safe_lines)

    @classmethod
    def sanitize_response(cls, response: str) -> str:
        for pattern in cls._secret_patterns:
            if pattern.search(response):
                return "Sensitive output was blocked by the prompt-injection guard."

        fallback = "Information not found in the datasheets."
        stripped = response.strip()
        if fallback in stripped and stripped != fallback:
            cleaned = stripped.replace(fallback + ":", "").replace(fallback, "").strip()
            if cleaned:
                return cleaned
        cleaned = re.sub(
            r"\n*Information not found in the datasheets[:.]?\s*$",
            "",
            stripped,
            flags=re.IGNORECASE,
        ).strip()
        if cleaned:
            return cleaned
        return response


# ---------------------------------------------------------------------------
# PDF Parsing – text + table extraction
# ---------------------------------------------------------------------------
def extract_pdf_content(pdf_path: str) -> list[Document]:
    """Extract text and preserve table structures from a single PDF."""
    documents: list[Document] = []
    filename = os.path.basename(pdf_path)
    doc = fitz.open(pdf_path)

    for page_num, page in enumerate(doc, start=1):
        blocks: list[str] = []

        # --- regular text ---
        text = page.get_text("text").strip()
        if text:
            blocks.append(text)

        # --- tables (PyMuPDF ≥ 1.23) ---
        try:
            tables = page.find_tables()
            for table in tables:
                # Render each table as a Markdown-style grid for the LLM
                header = table.header.names
                rows = table.extract()
                md_lines = [
                    "| " + " | ".join(str(c) if c else "" for c in header) + " |",
                    "| " + " | ".join("---" for _ in header) + " |",
                ]
                for row in rows:
                    md_lines.append(
                        "| " + " | ".join(str(c) if c else "" for c in row) + " |"
                    )
                blocks.append("\n".join(md_lines))
        except Exception:
            # Graceful fallback if table detection is unavailable
            pass

        full_content = "\n\n".join(blocks)
        if full_content.strip():
            documents.append(
                Document(
                    page_content=full_content,
                    metadata={"source": filename, "page": page_num},
                )
            )

    doc.close()
    return documents


def load_all_datasheets(source_path: str) -> list[Document]:
    """Load and parse every PDF in *source_path*."""
    all_docs: list[Document] = []
    for name in sorted(os.listdir(source_path)):
        if name.lower().endswith(".pdf"):
            all_docs.extend(extract_pdf_content(os.path.join(source_path, name)))
    return all_docs


def chunk_documents(documents: list[Document]) -> list[Document]:
    """Split documents into overlapping chunks suitable for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


# ---------------------------------------------------------------------------
# RAG Pipeline
# ---------------------------------------------------------------------------
class RAGPipeline:
    """End-to-end Retrieval-Augmented Generation pipeline."""

    TECHNICAL_SYNONYMS = {
        "i2c": [
            "i2c",
            "inter integrated circuit",
            "sda scl bus",
            "i2c address communication protocol",
        ],
        "spi": [
            "spi",
            "serial peripheral interface",
            "sclk sdi sdo cs",
            "spi communication protocol",
        ],
        "uart": [
            "uart",
            "serial interface tx rx",
            "universal asynchronous receiver transmitter",
        ],
        "uwb": [
            "uwb",
            "ultra wideband",
            "precision finding proximity",
        ],
        "airtag": [
            "airtag",
            "airtag device overview",
            "airtag support handling ultra wideband",
        ],
        "imu": [
            "imu",
            "inertial measurement unit",
            "accelerometer gyroscope motion sensor",
        ],
        "mpu-6050": [
            "mpu-6050",
            "mpu 6050",
            "motion processing unit mpu-6050",
        ],
        "mpu-6000": [
            "mpu-6000",
            "mpu 6000",
            "motion processing unit mpu-6000",
        ],
    }

    def __init__(self):
        self._ensure_local_provider_available()
        self.chroma_settings = self._build_chroma_settings()
        self.embeddings = self._build_embeddings()
        self.llm = self._build_llm()
        self.vector_store = None
        self.bm25_retriever: BM25Retriever | None = None
        self.ensemble_retriever: EnsembleRetriever | None = None
        self._chunks: list[Document] = []

    def _ensure_local_provider_available(self):
        provider = rag_cfg.get("provider", "ollama")
        embedding_provider = rag_cfg.get("embedding_provider") or provider

        if provider != "ollama" and embedding_provider != "ollama":
            return

        ollama_cfg = rag_cfg["ollama"]
        health_url = f"{ollama_cfg['base_url'].rstrip('/')}/api/tags"

        try:
            with urllib_request.urlopen(health_url, timeout=3) as response:
                if response.status >= 400:
                    raise RuntimeError
        except (urllib_error.URLError, RuntimeError):
            raise RuntimeError(
                "Ollama is configured as the default provider, but it is not "
                f"reachable at {ollama_cfg['base_url']}. Start Ollama with "
                "'ollama serve' and pull the required models, or switch "
                "provider=openai in config.yaml."
            )

    def _build_embeddings(self):
        provider = rag_cfg.get("embedding_provider") or rag_cfg.get("provider", "ollama")

        if provider == "ollama":
            ollama_cfg = rag_cfg["ollama"]
            return OllamaEmbeddings(
                model=ollama_cfg["embedding_model"],
                base_url=ollama_cfg["base_url"],
            )

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required when embedding_provider=openai")

        return OpenAIEmbeddings(
            model=rag_cfg["openai"]["embedding_model"],
            openai_api_key=api_key,
        )

    @staticmethod
    def _build_chroma_settings() -> Settings:
        return Settings(
            anonymized_telemetry=False,
            chroma_product_telemetry_impl="chroma_telemetry.NoOpProductTelemetryClient",
            chroma_telemetry_impl="chroma_telemetry.NoOpProductTelemetryClient",
        )

    def _build_llm(self):
        provider = rag_cfg.get("provider", "ollama")

        if provider == "ollama":
            ollama_cfg = rag_cfg["ollama"]
            return ChatOllama(
                model=ollama_cfg["llm_model"],
                base_url=ollama_cfg["base_url"],
                temperature=0,
            )

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required when provider=openai")

        return ChatOpenAI(
            model=rag_cfg["openai"]["llm_model"],
            openai_api_key=api_key,
            temperature=0,
        )

    @staticmethod
    def _get_vector_store_backend() -> str:
        return rag_cfg.get("vector_store_backend", "chroma").lower()

    @staticmethod
    def _get_vector_store_path() -> str:
        backend = RAGPipeline._get_vector_store_backend()
        path_map = rag_cfg.get("vector_store_paths", {})
        if backend in path_map:
            return path_map[backend]
        return rag_cfg.get("vector_db_path", "./vector_store/chroma_db")

    def _create_vector_store(self, documents: list[Document]):
        backend = self._get_vector_store_backend()
        store_path = self._get_vector_store_path()

        if backend == "faiss":
            os.makedirs(store_path, exist_ok=True)
            vector_store = FAISS.from_documents(documents, self.embeddings)
            vector_store.save_local(store_path)
            return vector_store

        if backend == "chroma":
            return Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                client_settings=self.chroma_settings,
                persist_directory=store_path,
            )

        raise ValueError(f"Unsupported vector_store_backend: {backend}")

    def _load_vector_store(self):
        backend = self._get_vector_store_backend()
        store_path = self._get_vector_store_path()

        if backend == "faiss":
            return FAISS.load_local(
                store_path,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )

        if backend == "chroma":
            return Chroma(
                persist_directory=store_path,
                client_settings=self.chroma_settings,
                embedding_function=self.embeddings,
            )

        raise ValueError(f"Unsupported vector_store_backend: {backend}")

    # ---- Ingestion --------------------------------------------------------

    def ingest(self, source_path: str | None = None):
        """Parse PDFs, chunk, and build both the vector store & BM25 index."""
        source_path = source_path or rag_cfg["pdf_source_path"]

        raw_docs = load_all_datasheets(source_path)
        self._chunks = chunk_documents(raw_docs)

        if not self._chunks:
            raise ValueError(f"No content extracted from PDFs in {source_path}")

        # -- Vector store (semantic) --
        self.vector_store = self._create_vector_store(self._chunks)

        # -- BM25 keyword index --
        self._build_hybrid_retriever()

        print(f"✓ Ingested {len(self._chunks)} chunks from {source_path}")

    def load_existing(self, source_path: str | None = None):
        """Load a persisted ChromaDB store and rebuild the BM25 index."""
        self.vector_store = self._load_vector_store()
        # BM25 is in-memory only → rebuild from source PDFs
        source_path = source_path or rag_cfg["pdf_source_path"]
        if os.path.isdir(source_path):
            raw_docs = load_all_datasheets(source_path)
            self._chunks = chunk_documents(raw_docs)
            self._build_hybrid_retriever()
        else:
            # Fall back to semantic-only if source PDFs are missing
            self.ensemble_retriever = None

    # ---- Retriever setup --------------------------------------------------

    def _build_hybrid_retriever(self):
        """Combine ChromaDB semantic retriever with BM25 keyword retriever."""
        top_k = rag_cfg["retrieval_settings"]["top_k"]

        semantic_retriever = self.vector_store.as_retriever(
            search_kwargs={"k": top_k}
        )

        self.bm25_retriever = BM25Retriever.from_documents(self._chunks)
        self.bm25_retriever.k = top_k

        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[semantic_retriever, self.bm25_retriever],
            weights=[0.5, 0.5],
        )

    # ---- Query chain ------------------------------------------------------

    def _get_retriever(self):
        search_type = rag_cfg["retrieval_settings"].get("search_type", "hybrid")
        top_k = rag_cfg["retrieval_settings"]["top_k"]

        if search_type == "hybrid" and self.ensemble_retriever:
            return self.ensemble_retriever
        if search_type == "keyword" and self.bm25_retriever:
            return self.bm25_retriever
        # default: semantic
        return self.vector_store.as_retriever(search_kwargs={"k": top_k})

    @staticmethod
    def _format_docs(docs: list[Document]) -> str:
        return "\n\n---\n\n".join(
            f"[Source: {d.metadata.get('source', '?')}, "
            f"Page {d.metadata.get('page', '?')}]\n{d.page_content}"
            for d in docs
        )

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-z0-9_+-]+", text.lower())

    @classmethod
    def _doc_relevance_score(cls, question: str, doc: Document) -> tuple[int, int, int]:
        question_tokens = cls._tokenize(question)
        token_counts = Counter(question_tokens)
        joined_question = " ".join(question_tokens)

        content = doc.page_content.lower()
        content_tokens = set(cls._tokenize(doc.page_content))

        overlap_score = sum(
            weight for token, weight in token_counts.items() if token in content_tokens
        )
        phrase_score = 0

        if "i2c" in joined_question and "i2c" in content:
            phrase_score += 5
            if "i2c communications protocol" in content:
                phrase_score += 10
            if "i2c bus" in content:
                phrase_score += 6
        if "address" in joined_question and "address" in content:
            phrase_score += 5
        if "i2c address" in joined_question and "i2c address" in content:
            phrase_score += 12
        if joined_question.strip() == "i2c" and any(
            term in content for term in ("slave address", "communications protocol", "start (s)", "stop (p)")
        ):
            phrase_score += 10
        if joined_question.strip() == "uart" and "uart" in content:
            phrase_score += 10
        if joined_question.strip() == "spi" and "spi" in content:
            phrase_score += 10
        if "slave address" in content:
            phrase_score += 8
        if "ad0" in content:
            phrase_score += 4

        exact_bonus = 10 if any(term in content for term in ("b1101000", "b1101001", "b110100x")) else 0

        return (overlap_score + phrase_score + exact_bonus, phrase_score, -len(doc.page_content))

    @classmethod
    def _rerank_docs(cls, question: str, docs: list[Document]) -> list[Document]:
        unique_docs: list[Document] = []
        seen_keys: set[tuple[str, int, str]] = set()

        for doc in docs:
            key = (
                str(doc.metadata.get("source", "")),
                int(doc.metadata.get("page", -1)),
                doc.page_content[:200],
            )
            if key not in seen_keys:
                seen_keys.add(key)
                unique_docs.append(doc)

        reranked = sorted(unique_docs, key=lambda doc: cls._doc_relevance_score(question, doc), reverse=True)
        top_k = rag_cfg["retrieval_settings"].get("top_k", 5)
        return reranked[:top_k]

    @classmethod
    def _safe_context_from_docs(cls, docs: list[Document]) -> str:
        return PromptInjectionGuard.sanitize_context(cls._format_docs(docs))

    @staticmethod
    def _build_search_query(question: str) -> str:
        topic = RAGPipeline._extract_topic(question)
        if topic and RAGPipeline._is_technical_topic(topic):
            return topic
        if topic:
            return f"{topic} overview features specifications usage support details"

        stripped = question.strip()
        tokens = RAGPipeline._tokenize(stripped)
        if 0 < len(tokens) <= 3 and not any(char in stripped for char in "?:"):
            return f"{stripped} overview features specifications usage support details"

        return stripped

    @classmethod
    def _expand_query_variants(cls, question: str) -> list[str]:
        stripped = question.strip()
        base_query = cls._build_search_query(stripped)
        topic = (cls._extract_topic(stripped) or stripped).strip().lower()

        variants: list[str] = [base_query]

        if topic in cls.TECHNICAL_SYNONYMS:
            variants.extend(cls.TECHNICAL_SYNONYMS[topic])

        tokens = cls._tokenize(stripped)
        if "address" in tokens:
            variants.append(f"{base_query} register address slave address pin configuration")
        if "protocol" in tokens or topic in {"i2c", "spi", "uart"}:
            variants.append(f"{base_query} communication protocol timing interface pins")
        if cls._is_broad_topic_question(stripped):
            variants.append(f"{topic} overview features specifications usage details")

        seen: set[str] = set()
        deduped: list[str] = []
        for variant in variants:
            normalized = " ".join(variant.split()).strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                deduped.append(normalized)
        return deduped

    @classmethod
    def _merge_candidate_docs(cls, doc_groups: list[list[Document]]) -> list[Document]:
        merged: list[Document] = []
        seen_keys: set[tuple[str, int, str]] = set()

        for docs in doc_groups:
            for doc in docs:
                key = (
                    str(doc.metadata.get("source", "")),
                    int(doc.metadata.get("page", -1)),
                    doc.page_content[:200],
                )
                if key not in seen_keys:
                    seen_keys.add(key)
                    merged.append(doc)
        return merged

    @staticmethod
    def _extract_topic(question: str) -> str | None:
        stripped = question.strip()
        lowered = stripped.lower()

        conversational_prefixes = (
            "talk about ",
            "talk me about ",
            "talk more about ",
            "tell me about ",
            "tell me more about ",
            "more about ",
            "explain ",
            "what about ",
        )
        for prefix in conversational_prefixes:
            if lowered.startswith(prefix):
                topic = stripped[len(prefix):].strip(" ?.!:")
                return topic or None

        tokens = RAGPipeline._tokenize(stripped)
        if 0 < len(tokens) <= 3 and not any(char in stripped for char in "?:"):
            return stripped.strip(" ?.!:") or None

        return None

    @staticmethod
    def _is_technical_topic(topic: str) -> bool:
        tokens = RAGPipeline._tokenize(topic)
        if not tokens:
            return False

        if len(tokens) == 1:
            raw = topic.strip()
            if len(raw) <= 6 and raw.upper() == raw:
                return True

        technical_terms = {"i2c", "spi", "uart", "uwb", "ble", "imu", "mpu-6050", "mpu-6000"}
        return all(token in technical_terms for token in tokens)

    @staticmethod
    def _is_broad_topic_question(question: str) -> bool:
        lowered = question.strip().lower()
        if lowered.startswith((
            "talk about ",
            "talk me about ",
            "talk more about ",
            "tell me about ",
            "tell me more about ",
            "more about ",
            "explain ",
            "what about ",
        )):
            return True

        tokens = RAGPipeline._tokenize(question)
        return 0 < len(tokens) <= 3 and not any(char in question for char in "?:")

    def query_sensor_info(
        self,
        question: str,
        prediction_payload: dict | None = None,
    ) -> str:
        """Answer a hardware-related question using only datasheet context."""
        if self.vector_store is None:
            raise RuntimeError("Pipeline not initialised. Call ingest() or load_existing() first.")

        blocked_reason = PromptInjectionGuard.validate_question(question)
        if blocked_reason:
            return blocked_reason

        broad_topic_question = self._is_broad_topic_question(question)
        broad_topic = self._extract_topic(question) or question.strip()
        ranking_query = broad_topic if broad_topic_question else question
        retrieval_query = ReasoningPipeline.build_retrieval_query(question, prediction_payload)

        retriever = self._get_retriever()
        search_queries = self._expand_query_variants(retrieval_query)
        candidate_doc_groups = [retriever.invoke(search_query) for search_query in search_queries]
        retrieved_docs = self._merge_candidate_docs(candidate_doc_groups)
        reranked_docs = self._rerank_docs(ranking_query, retrieved_docs)

        if not reranked_docs:
            return "Information not found in the datasheets."

        if broad_topic_question and self._doc_relevance_score(ranking_query, reranked_docs[0])[0] < 1:
            return "Information not found in the datasheets."

        context = self._safe_context_from_docs(reranked_docs)

        return ReasoningPipeline.generate_grounded_answer(
            llm=self.llm,
            question=question,
            context=context,
            prediction_payload=prediction_payload,
            broad_topic=broad_topic if broad_topic_question else None,
            sanitize_response=PromptInjectionGuard.sanitize_response,
        )


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------
_pipeline: RAGPipeline | None = None


def query_sensor_info(question: str, prediction_payload: dict | None = None) -> str:
    """One-call interface: initialises the pipeline on first use, then queries."""
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
        _pipeline.ingest()
    return _pipeline.query_sensor_info(question, prediction_payload=prediction_payload)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pipeline = RAGPipeline()

    if "--ingest" in sys.argv:
        pipeline.ingest()
    else:
        pipeline.load_existing()

    # Interactive loop
    while True:
        q = input("\nQuestion (or 'quit'): ").strip()
        if q.lower() in ("quit", "exit", "q"):
            break
        print(f"\n{pipeline.query_sensor_info(q)}")
