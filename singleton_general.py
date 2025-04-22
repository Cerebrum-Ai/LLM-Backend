from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ChatMessageHistory
from threading import Lock
import uuid
from datetime import datetime

class ChatSession:
    def __init__(self):
        self.history = ChatMessageHistory()
        self.last_accessed = datetime.now()
        self.session_id = str(uuid.uuid4())

class LLMManager:
    _instance = None
    _multimodal_llm = None
    _medical_llm = None
    _sessions = {}
    _session_lock = Lock()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        if LLMManager._instance is not None:
            raise Exception("Use get_instance() instead")
        
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        n_gpu_layers = -1
        n_batch = 2048
        n_ctx = 2048
        self._multimodal_llm = LlamaCpp(
            model_path=r"/content/LLM-Backend/Bio-Medical-MultiModal-Llama-3-8B-V1.i1-Q4_K_M.gguf",
            n_gpu_layers=n_gpu_layers,
            n_batch=n_batch,
            callback_manager=callback_manager,
            n_ctx=n_ctx,
            verbose=True,
        )
        self._medical_llm = LlamaCpp(
            model_path=r"/content/LLM-Backend/phi-2.Q5_K_M.gguff",
            n_gpu_layers=n_gpu_layers,
            n_batch=n_batch,
            callback_manager=callback_manager,
            n_ctx=n_ctx,
            verbose=True,  # Verbose is required to pass to the callback manager
        )

    @property
    def llm(self):
        """Primary LLM for the application"""
        return self._multimodal_llm

    def create_session(self):
        """Create a new chat session and return its ID"""
        with self._session_lock:
            session = ChatSession()
            self._sessions[session.session_id] = session
            return session.session_id

    def get_session(self, session_id: str) -> ChatSession:
        """Get a session by ID"""
        with self._session_lock:
            session = self._sessions.get(session_id)
            if session:
                session.last_accessed = datetime.now()
            return session

    def clear_session(self, session_id: str) -> bool:
        """Clear the history of a session"""
        with self._session_lock:
            session = self._sessions.get(session_id)
            if session:
                session.history.clear()
                return True
            return False

    def delete_session(self, session_id: str) -> bool:
        """Delete a session"""
        with self._session_lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                return True
            return False

    def add_to_history(self, session_id: str, user_message: str, ai_message: str) -> bool:
        """Add a message exchange to the session history"""
        with self._session_lock:
            session = self._sessions.get(session_id)
            if session:
                session.history.add_user_message(user_message)
                session.history.add_ai_message(ai_message)
                return True
            return False

    @property
    def multimodal_llm(self):
        return self._multimodal_llm
    @property
    def medical_llm(self):
        return self._medical_llm


class VectorDBManager:
    _instance = None
    _vector_store = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        if VectorDBManager._instance is not None:
            raise Exception("Use get_instance() instead")
        
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        loader = TextLoader("medical_data.csv")
        docs = loader.load()

        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1100, chunk_overlap=50)
        all_splits = text_splitter.split_documents(docs)

        self._vector_store = Chroma(
            collection_name="medical_data",
            embedding_function=embeddings,
            persist_directory="./chroma_langchain_db",
        )
        _ = self._vector_store.add_documents(documents=all_splits)

    @property
    def vector_store(self):
        return self._vector_store