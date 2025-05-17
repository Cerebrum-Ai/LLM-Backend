import os
from langchain_openai import ChatOpenAI  # Changed from langchain_openrouter
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
import threading
import base64
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate

class OpenRouterLLMManager:
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = OpenRouterLLMManager()
        return cls._instance
    
    def __init__(self):
        self.llm = None
        self.medical_llm = None
        self.sessions = {}
        self.is_inferencing = False
        self._inference_lock = threading.Lock()
        self.initialize()
    
    def initialize(self):
        """Initialize the OpenRouter LLM"""
        try:
            api_key = os.environ.get("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY environment variable not set")
            
            # Initialize main LLM with vision capabilities
            model_name = os.environ.get("OPENROUTER_MODEL", "meta-llama/llama-4-maverick:free")
            medical_model_name_reasoning = os.environ.get("OPENROUTER_MODEL_final","meta-llama/llama-4-scout:free" )
            # Configure the LLM to work with the pipe operator using OpenAI's ChatOpenAI with OpenRouter base URL
            self.llm = ChatOpenAI(
                model=model_name,
                openai_api_key=api_key,
                openai_api_base="https://openrouter.ai/api/v1",
                temperature=0.7,
                max_tokens=196,
                
            )
            
            # Initialize medical LLM (can be the same model or different)
            medical_model_name_reasoning = os.environ.get("OPENROUTER_MEDICAL_final", medical_model_name_reasoning)
            self.medical_llm = ChatOpenAI(
                model=medical_model_name_reasoning,
                openai_api_key=api_key,
                openai_api_base="https://openrouter.ai/api/v1",
                temperature=0.5,  # Lower temperature for medical responses
                max_tokens=512,
            )
            
            print(f"OpenRouter LLM initialized with model: {model_name}")
            print(f"OpenRouter Medical LLM initialized with model: {medical_model_name_reasoning}")
            return True
        except Exception as e:
            print(f"Error initializing OpenRouter LLM: {str(e)}")
            return False
    
    def create_session(self):
        """Create a new session ID"""
        import uuid
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = []
        return session_id
    
    def delete_session(self, session_id):
        """Delete a session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
    
    def start_inference(self):
        """Mark that inference is starting - used to pause heartbeats"""
        with self._inference_lock:
            self.is_inferencing = True
    
    def end_inference(self):
        """Mark that inference is complete - used to resume heartbeats"""
        with self._inference_lock:
            self.is_inferencing = False
    
    def _process_image(self, image):
        """Process image data for OpenRouter models"""
        if not image:
            return None
            
        # Handle different image formats
        if isinstance(image, str):
            # If it's a URL
            if image.startswith('http'):
                return image  # Return URL as is
            elif image.startswith('data:image'):
                return image  # Return base64 data URL as is
            else:
                # Assume it's a file path
                try:
                    with open(image, 'rb') as img_file:
                        img_data = base64.b64encode(img_file.read()).decode('utf-8')
                        return f"data:image/jpeg;base64,{img_data}"
                except Exception as e:
                    print(f"Error processing image file: {str(e)}")
                    return None
        elif hasattr(image, 'read'):  # File-like object
            try:
                img_data = base64.b64encode(image.read()).decode('utf-8')
                # Reset file pointer
                image.seek(0)
                return f"data:image/jpeg;base64,{img_data}"
            except Exception as e:
                print(f"Error processing image file: {str(e)}")
                return None
        return None
    def _process_llm_response(self, response):
        """Process LLM response to ensure consistent string output"""
        if hasattr(response, 'content'):
            return response.content
        return str(response)

    def process_chain_response(self, chain_response):
        """Process chain response to handle both direct and AIMessage responses"""
        try:
            if hasattr(chain_response, 'content'):
                return chain_response.content
            elif isinstance(chain_response, str):
                return chain_response
            return str(chain_response)
        except Exception as e:
            print(f"Error processing chain response: {str(e)}")
            return str(chain_response)

    def init_llm_input(self, question, image=None, ml_results=None):
        """Process input with multimodal LLM"""
        try:
            response = self.llm.invoke([
                SystemMessage(content="You are a medical diagnosis assistant."),
                HumanMessage(content=question)
            ])
            # Ensure we return a string, not an AIMessage object
            return self.process_chain_response(response)
        except Exception as e:
            print(f"Error in init_llm_input: {str(e)}")
            return str(e)

    def post_llm_input(self, initial_diagnosis, question, context, ml_results=None):
        """Process follow-up with context"""
        try:
            context_text = "\n".join([doc.page_content for doc in context]) if context else ""
            response = self.medical_llm.invoke([
                SystemMessage(content="You are a medical diagnosis assistant."),
                HumanMessage(content=f"Initial Diagnosis: {initial_diagnosis}\nContext: {context_text}\nQuestion: {question}")
            ])
            # Ensure we return a string, not an AIMessage object
            return self.process_chain_response(response)
        except Exception as e:
            print(f"Error in post_llm_input: {str(e)}")
            return str(e)