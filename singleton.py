from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.chat_message_histories import ChatMessageHistory
from threading import Lock
import uuid
from datetime import datetime
import os
import torch
import pickle
import csv
import time

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
    _multimodal_llm_cache_path = "multimodal_llm.pkl"
    _medical_llm_cache_path = "medical_llm.pkl"
    _inference_lock = Lock()
    _is_inferencing = False
    

    def start_inference(self):
        """Pause heartbeats during inference"""
        with self._inference_lock:
            self._is_inferencing = True

    def end_inference(self):
        """Resume heartbeats after inference"""
        with self._inference_lock:
            self._is_inferencing = False

    @property
    def is_inferencing(self):
        """Check if the model is currently inferencing"""
        with self._inference_lock:
            return self._is_inferencing

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        if LLMManager._instance is not None:
            raise Exception("Use get_instance() instead")
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        n_gpu_layers =  25
        n_batch = 512
        n_ctx = 2048
        llm_params = {
            'n_gpu_layers': n_gpu_layers,
            'n_batch': n_batch,
            'n_ctx': n_ctx,
            'verbose': False,
            'use_mlock': True,
            'use_mmap': True,
            'f16_kv': True,
            'seed': -1,
            'top_k': 40,     # Reduced for faster sampling
            'n_threads': (os.cpu_count())
        }
        model_kwargs = {
            'tensor_split': None,
            'rope_scaling': {'type': 'linear', 'factor': 0.5}
        }

        if os.path.exists(self._multimodal_llm_cache_path):
            with open(self._multimodal_llm_cache_path, 'rb') as f:
                self._multimodal_llm = pickle.load(f)
            print("Loaded multimodal LLM from cache")
        else:    
            self._multimodal_llm = LlamaCpp(
                model_path=r"Bio-Medical-MultiModal-Llama-3-8B-V1.Q4_K_M.gguf",
                callback_manager=callback_manager,
                max_tokens=35,
                temperature=0.5,
                top_p=0.95,
                repeat_penalty=1.1,
                model_kwargs=model_kwargs,
                **llm_params
                
            )
            with open(self._multimodal_llm_cache_path, 'wb') as f:
                pickle.dump(self._multimodal_llm, f)
            print("Initialized and saved multimodal LLM to cache")

        if os.path.exists(self._medical_llm_cache_path):
            with open(self._medical_llm_cache_path, 'rb') as f:
                self._medical_llm = pickle.load(f)
            print("Loaded medical LLM from cache")
        else:
            
            self._medical_llm = LlamaCpp(
                model_path=r"phi-2.Q5_K_M.gguf",
                callback_manager=callback_manager,
                max_tokens=75,
                temperature=0.3,
                top_p=0.95,
                repeat_penalty=1.1,
                model_kwargs=model_kwargs,
                **llm_params
                  
            )
            with open(self._medical_llm_cache_path, 'wb') as f:
                pickle.dump(self._medical_llm, f)
            print("Initialized and saved medical LLM to cache")


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
    _embeddings = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        if VectorDBManager._instance is not None:
            raise Exception("Use get_instance() instead")
        if torch.cuda.is_available():
            print("CUDA is available")
            self._embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2", model_kwargs={'device': 'cuda'})
        else:
            print("CUDA is not available")
            self._embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

        embeddings_path = "medical_data_embeddings.pkl"
        all_splits_path = "medical_data_documents.pkl"

        if os.path.exists("chroma_langchain_db"):
            # Load existing Chroma DB
            self._vector_store = Chroma(
                collection_name="medical_data",
                embedding_function=self._embeddings,
                persist_directory="./chroma_langchain_db"
            )
            print("Loaded existing Chroma DB")
        else:
            # Check if embeddings and documents are pre-calculated
            if os.path.exists(embeddings_path) and os.path.exists(all_splits_path):
                # Load pre-calculated embeddings and documents
                with open(embeddings_path, "rb") as f:
                    _embeddings = pickle.load(f)
                with open(all_splits_path, "rb") as f:
                    all_splits = pickle.load(f)

                # Create Chroma DB from pre-calculated embeddings
                self._vector_store = Chroma(
                    embedding_function=self._embeddings,
                    collection_name="medical_data",
                    persist_directory="./chroma_langchain_db"
                )
                _ = self._vector_store.add_documents(documents=all_splits)
                print("Loaded Chroma DB from pre-calculated embeddings")
            else:
                # Calculate embeddings and create Chroma DB
                loader = TextLoader("medical_data.csv")
                docs = loader.load()

                text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1100, chunk_overlap=50)
                all_splits = text_splitter.split_documents(docs)

                self._vector_store = Chroma(
                    embedding_function=self._embeddings,
                    collection_name="medical_data",
                    persist_directory="./chroma_langchain_db"
                )
                _ = self._vector_store.add_documents(documents=all_splits)
                # Persist embeddings and documents for future use
                with open(embeddings_path, "wb") as f:
                    pickle.dump(self._embeddings, f)
                with open(all_splits_path, "wb") as f:
                    pickle.dump(all_splits, f)
                print("Created Chroma DB and persisted embeddings")

        

    @property
    def vector_store(self):
        return self._vector_store

   

    
