from langchain_community.cache import InMemoryCache
from langchain_core.globals import set_llm_cache
import base64
from singleton import LLMManager
import requests
from langchain_core.prompts import PromptTemplate

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
            if data_type == "audio" and "detected_emotion" in result:
                ml_context_parts.append(f"Audio analysis detected emotional state: {result['detected_emotion']}")
            # Add other ML result types as they become available
        ml_context = " ".join(ml_context_parts)
    
    # Pre-compile templates for better performance
    if image is None:
        template = """Question: {question}, Additional attached data: {not_none_keys}
{ml_context}

ATTENTION: THIS IS A LIST-ONLY RESPONSE.
Instructions:
1. Identify potential diseases based on the question and any ML analysis results.
2. List ONLY disease names (maximum 5).
3. Provide the disease names as a comma-separated list.
4. Do not include any introductory text, explanations, or additional information.
5. Do not include the question, instructions, or attention text in the answer.
6. Limit the response to 10 words or less.
7. All questions are purely medical and require only the disease names.

Answer: """
    else:
        template = """Question: {question}, image data: {image} Additional attached data: {not_none_keys}
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

Answer: """
    
    prompt = PromptTemplate.from_template(template)
    llm_chain = prompt | llm_manager.llm
    
    try:
        llm_manager.start_inference()  # Pause heartbeats during inference
        return llm_chain.invoke({
            "question": question,
            "image": image,
            "not_none_keys": not_none_keys,
            "ml_context": ml_context
        })
    finally:
        llm_manager.end_inference()  # Resume heartbeats
   

def post_llm_input(initial_diagnosis, question, context, not_none_keys_data=None, ml_results=None):
    """Process follow-up with context"""
    llm_manager = LLMManager.get_instance()
    
    # Extract ML analysis results from data for LLM context
    ml_context_parts = []
    if not_none_keys_data:
        # Extract audio emotion if available
        if isinstance(not_none_keys_data.get("audio"), dict) and "emotion" in not_none_keys_data["audio"]:
            emotion = not_none_keys_data["audio"]["emotion"].get("detected_emotion", "unknown")
            ml_context_parts.append(f"Audio analysis detected emotional state: {emotion}")
        
        # Add other ML result types as they become available
        # For example: image analysis, gait analysis, typing pattern analysis
    
    if ml_results:
        for data_type, result in ml_results.items():
            if data_type == "audio" and "detected_emotion" in result:
                ml_context_parts.append(f"Audio analysis detected emotional state: {result['detected_emotion']}")
            # Add other ML result types as they become available
    
    ml_context = "\n".join(ml_context_parts)
    
    template = """Question: {question}
Additional attached data: {not_none_keys_data}
Context: {context}
Initial Diagnosis: {initial_diagnosis}
ML Analysis: {ml_context}

Instructions: Provide a concise medical analysis following this exact format:

Diagnosis: [most relevant condition]
Symptoms: [list up to 5 key symptoms, comma-separated]
Treatment: [list up to 3 primary treatments, comma-separated]
Emotional State: [if detected in audio analysis]

1. Focus on the most relevant condition based on the question, context, initial diagnosis, and ML analysis.
2. List up to 5 key symptoms, separated by commas.
3. List up to 3 primary treatments, separated by commas.
4. If emotional state was detected (from audio analysis), include it if clinically relevant.
5. Keep the entire response under 50 words.
6. Omit all introductory text, context, and explanations.
7. Output ONLY the formatted sections as shown above.

Answer:
"""
    
    prompt = PromptTemplate.from_template(template)
    llm_chain = prompt | llm_manager.medical_llm
    
    try:
        llm_manager.start_inference()  # Pause heartbeats during inference
        return llm_chain.invoke({
            "question": question,
            "context": context,
            "initial_diagnosis": initial_diagnosis,
            "not_none_keys_data": not_none_keys_data,
            "ml_context": ml_context
        })
    finally:
        llm_manager.end_inference()  # Resume heartbeats