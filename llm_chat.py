from langchain_community.cache import InMemoryCache
from langchain_core.globals import set_llm_cache
import base64
from singleton import LLMManager
import requests
from langchain_core.prompts import PromptTemplate
import json

# Enable in-memory caching for repeated queries
set_llm_cache(InMemoryCache())

def init_llm_input(question, image=None, ml_results=None):
    """Process input with multimodal LLM"""
    llm_manager = LLMManager.get_instance()
    
    # Format ML results for LLM context if available
    ml_context = ""
    if ml_results:
        ml_context_parts = []
        for data_type, result in ml_results.items():
            if isinstance(result, dict):
                for k, v in result.items():
                    ml_context_parts.append(f"{data_type} {k}: {v}")
            else:
                ml_context_parts.append(f"{data_type}: {result}")
        ml_context = " | ".join(ml_context_parts)
    
    # Pre-compile templates for better performance
    if image is None:
        template = """Question: {question}
This is the typing,audio, and other data that has been analysed by ml models utilize  this to make the diagnosis more accurate:
{ml_context}


ATTENTION: THIS IS A LIST-ONLY RESPONSE.
Instructions:
1. Identify potential diseases based on the question, image data, and any ML analysis results.
2. List ONLY disease names (maximum 5).
3. Provide the disease names as a comma-separated list.
4. Do not include any introductory text, explanations, or additional information.
5. Do not include the question, instructions, or attention text in the answer.
6. Limit the response to 10 words or less.
7. All questions are purely medical and require only the disease names.
8. Never return empty,always return a diagnosis from the data

Answer: """
    else:
        template = """Question: {question}, image_url: {image_url}
This is the typing,audio, and other data that has been analysed by ml models utilize  this to make the diagnosis more accurate:
{ml_context}

ATTENTION: THIS IS A LIST-ONLY RESPONSE.
Instructions:
1. Identify potential diseases based on the question, image data, and any ML analysis results.
2. List ONLY disease names (maximum 5).
3. Provide the disease names as a comma-separated list.
4. Do not include any introductory text, explanations, or additional information.
5. Do not include the question, instructions, or attention text in the answer.
6. Limit the response to 10 words or less.
7. All questions are purely medical and require only the disease names.
8. Never return empty,always return a diagnosis from the data

Answer: """
    
    prompt = PromptTemplate.from_template(template)
    llm_chain = prompt | llm_manager.llm
    
    try:
        llm_manager.start_inference()  # Pause heartbeats during inference
        return llm_chain.invoke({
            "question": question,
            "image_url": image,
            "ml_context": ml_context
        })
    finally:
        llm_manager.end_inference()  # Resume heartbeats
   

def post_llm_input(initial_diagnosis, question, context, ml_results=None):
    """Process follow-up with context"""
    llm_manager = LLMManager.get_instance()
    
    # Format ML results for LLM context if available
    ml_context = ""
    ml_context_parts = []
    if ml_results:
        for data_type, result in ml_results.items():
            if isinstance(result, dict):
                for k, v in result.items():
                    ml_context_parts.append(f"{data_type} {k}: {v}")
            else:
                ml_context_parts.append(f"{data_type}: {result}")
        ml_context = " | ".join(ml_context_parts)

    template = """Question: {question}
Context: {context} ( data from a vector database regarding the initial diagnosis)
Initial Diagnosis: {initial_diagnosis} (made by a medical model)
This is the typing,audio, and other data that has been analysed by ml models utilize  this to make the diagnosis more accurate:
{ml_context}

Instructions: Provide a concise medical analysis following this exact format :

Diagnosis: [most relevant condition, always provide a diagnosis from the data never return empty]
Symptoms: [list up to 5 key symptoms, comma-separated]
Treatment: [list up to 3 primary treatments, comma-separated]
Emotional State: [if detected in audio analysis]

Dont add anything extra only list the above as per the data analysed and initial_diagnosis

Answer: """
    prompt = PromptTemplate.from_template(template)
    llm_chain = prompt | llm_manager.medical_llm
    try:
        llm_manager.start_inference()  # Pause heartbeats during inference
        return llm_chain.invoke({
            "question": question,
            "context": context,
            "initial_diagnosis": initial_diagnosis,
            "ml_context": ml_context
        })
    finally:
        llm_manager.end_inference()  # Resume heartbeats
