from langchain_community.cache import InMemoryCache
from langchain_core.globals import set_llm_cache
import base64
from singleton import LLMManager
import requests
from langchain_core.prompts import PromptTemplate
import json

# Enable in-memory caching for repeated queries
set_llm_cache(InMemoryCache())

def init_llm_input(question, image=None, not_none_keys=None, ml_results=None):
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
    
    # Format all attached data as pretty JSON for context
    attached_data_str = ""
    if not_none_keys:
        try:
            attached_data_str = json.dumps(not_none_keys, indent=2, ensure_ascii=False)
        except Exception:
            attached_data_str = str(not_none_keys)
    
    # Pre-compile templates for better performance
    if image is None:
        template = """Question: {question}
Attached Data:
{attached_data}
ML Context:
{ml_context}

Instructions: Based on the question and provided data, return ONLY the following fields in this order, with no extra text or explanation:

Diagnosis: [most relevant condition]
Symptoms: [list up to 5 key symptoms, comma-separated]
Treatment: [list up to 3 primary treatments, comma-separated]
Emotional State: [if detected in audio analysis]

Dont add anything extra only list the above as per the data analysed.

Answer: """
    else:
        template = """Question: {question}, image_url: {image_url} Additional attached data: {not_none_keys}
        ml_context: {ml_context}

ATTENTION: THIS IS A LIST-ONLY RESPONSE.
Instructions:
1. Identify potential diseases based on the question, image data, and any ML analysis results.
2. List ONLY disease names (maximum 5).
3. Provide the disease names as a comma-separated list.
4. Do not include any introductory text, explanations, or additional information.
5. Do not include the question, instructions, or attention text in the answer.
6. Limit the response to 10 words or less.
7. All questions are purely medical and require only the disease names.

Answer: """
    
    prompt = PromptTemplate.from_template(template)
    llm_chain = prompt | llm_manager.llm
    
    try:
        llm_manager.start_inference()  # Pause heartbeats during inference
        return llm_chain.invoke({
            "question": question,
            "image_url": image,
            "not_none_keys": not_none_keys,
            "ml_context": ml_context
        })
    finally:
        llm_manager.end_inference()  # Resume heartbeats
   

def post_llm_input(initial_diagnosis, question, context, not_none_keys_data=None, ml_results=None):
    """Process follow-up with context"""
    llm_manager = LLMManager.get_instance()
    import json
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
    # Format all attached data as pretty JSON for context
    attached_data_str = ""
    if not_none_keys_data:
        try:
            attached_data_str = json.dumps(not_none_keys_data, indent=2, ensure_ascii=False)
        except Exception:
            attached_data_str = str(not_none_keys_data)
    template = """Question: {question}
Attached Data:
{attached_data}
Context: {context}
Initial Diagnosis: {initial_diagnosis} ( give a bit of priority to this)
ML Analysis: {ml_context} 

Instructions: Provide a concise medical analysis following this exact format :

Diagnosis: [most relevant condition]
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
            "attached_data": attached_data_str,
            "ml_context": ml_context
        })
    finally:
        llm_manager.end_inference()  # Resume heartbeats
