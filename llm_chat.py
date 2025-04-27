from langchain_community.cache import InMemoryCache
from langchain_core.globals import set_llm_cache
import base64
from singleton import LLMManager
import requests
from langchain_core.prompts import PromptTemplate

# Enable in-memory caching for repeated queries
set_llm_cache(InMemoryCache())

def init_llm_input(question, image=None, not_none_keys=None, audio_context=None):
    """Process input with multimodal LLM"""
    llm_manager = LLMManager.get_instance()
    
    # Pre-compile templates for better performance
    if image is None and not audio_context: 
        template = """Question: {question}, Additional attached data: {not_none_keys}
ATTENTION: THIS IS A LIST-ONLY RESPONSE.
Instructions:
1. Identify potential diseases based on the question.
2. List ONLY disease names (maximum 5).
3. Provide the disease names as a comma-separated list.
4. Do not include any introductory text, explanations, or additional information.
5. Do not include the question, instructions, or attention text in the answer.
6. Limit the response to 10 words or less.
7. All questions are purely medical and require only the disease names.

Answer: """
    elif image is not None and not audio_context:
        template = """Question: {question}, image data: {image} Additional attached data: {not_none_keys}
ATTENTION: THIS IS A LIST-ONLY RESPONSE.
Instructions:
1. Identify potential diseases based on the question and image data.
2. List ONLY disease names (maximum 5).
3. Provide the disease names as a comma-separated list.
4. Do not include any introductory text, explanations, or additional information.
5. Do not include the question, instructions, or attention text in the answer.
6. Limit the response to 10 words or less.
7. All questions are purely medical and require only the disease names.

Answer: """
    elif audio_context and image is None:
        template = """Question: {question}, Additional attached data: {not_none_keys}
Audio analysis: {audio_context}

ATTENTION: THIS IS A LIST-ONLY RESPONSE.
Instructions:
1. Identify potential diseases based on the question and emotional state detected in audio.
2. List ONLY disease names (maximum 5).
3. Provide the disease names as a comma-separated list.
4. Do not include any introductory text, explanations, or additional information.
5. Do not include the question, instructions, or attention text in the answer.
6. Limit the response to 10 words or less.
7. All questions are purely medical and require only the disease names.
8. Consider that emotional states like 'sad' may indicate depression, while 'fear' might suggest anxiety disorders.

Answer: """
    else:
        template = """Question: {question}, image data: {image} Additional attached data: {not_none_keys}
Audio analysis: {audio_context}

ATTENTION: THIS IS A LIST-ONLY RESPONSE.
Instructions:
1. Identify potential diseases based on the question, image data, and emotional state detected in audio.
2. List ONLY disease names (maximum 5).
3. Provide the disease names as a comma-separated list.
4. Do not include any introductory text, explanations, or additional information.
5. Do not include the question, instructions, or attention text in the answer.
6. Limit the response to 10 words or less.
7. All questions are purely medical and require only the disease names.
8. Consider that emotional states like 'sad' may indicate depression, while 'fear' might suggest anxiety disorders.

Answer: """
    
    prompt = PromptTemplate.from_template(template)
    llm_chain = prompt | llm_manager.llm

    try:
        llm_manager.start_inference()  # Pause heartbeats during inference
        return llm_chain.invoke({
            "question": question,
            "image": image,
            "not_none_keys": not_none_keys,
            "audio_context": audio_context
        })
    finally:
        llm_manager.end_inference()  # Resume heartbeats
    
   

def post_llm_input(initial_diagnosis, question, context, not_none_keys_data=None, audio_context=None):
    """Process follow-up with context"""
    llm_manager = LLMManager.get_instance()
    
    # Add audio context to template if available
    if audio_context:
        template = """Question: {question}
Additional attached data: {not_none_keys_data}
Context: {context}
Initial Diagnosis: {initial_diagnosis}
Audio Analysis: {audio_context}

Instructions: Provide a concise medical analysis following this exact format:

Diagnosis: [most relevant condition]
Symptoms: [list up to 5 key symptoms, comma-separated]
Treatment: [list up to 3 primary treatments, comma-separated]
Emotional State: [emotional state from audio, if relevant to diagnosis]

1. Focus on the most relevant condition based on the question, context, initial diagnosis, and patient's emotional state.
2. List up to 5 key symptoms, separated by commas.
3. List up to 3 primary treatments, separated by commas.
4. If emotional state from audio is clinically relevant (e.g., depression), include it in analysis.
5. Keep the entire response under 50 words.
6. Omit all introductory text, context, and explanations.
7. Output ONLY the formatted sections as shown above.

Answer:
"""
    else:
        template = """Question: {question}
Additional attached data: {not_none_keys_data}
Context: {context}
Initial Diagnosis: {initial_diagnosis}

Instructions: Provide a concise medical analysis following this exact format:

Diagnosis: [most relevant condition]
Symptoms: [list up to 5 key symptoms, comma-separated]
Treatment: [list up to 3 primary treatments, comma-separated]

1. Focus on the most relevant condition based on the question, context, and initial diagnosis.
2. List up to 5 key symptoms, separated by commas.
3. List up to 3 primary treatments, separated by commas.
4. Keep the entire response under 50 words.
5. Omit all introductory text, context, and explanations.
6. Output ONLY the formatted sections as shown above.

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
            "audio_context": audio_context
        })
    finally:
        llm_manager.end_inference()  # Resume heartbeats

        