from langchain_community.cache import InMemoryCache
from langchain_core.globals import set_llm_cache
import base64
from singleton import LLMManager
import requests
from langchain_core.prompts import PromptTemplate

# Enable in-memory caching for repeated queries
set_llm_cache(InMemoryCache())

def init_llm_input(question, image=None):
    """Process input with multimodal LLM"""
    llm_manager = LLMManager.get_instance()
    
    # Pre-compile templates for better performance
    if image is None:
        template = """Question: {question}
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
    else:
        template = """Question: {question} and image data: {image}
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
    
    prompt = PromptTemplate.from_template(template)
    llm_chain = prompt | llm_manager.llm

    try:
        llm_manager.start_inference()  # Pause heartbeats during inference
        return llm_chain.invoke({
            "question": question,
            "image": image
        })
    finally:
        llm_manager.end_inference()  # Resume heartbeats
    
   

def post_llm_input(initial_diagnosis, question, context):
    """Process follow-up with context"""
    llm_manager = LLMManager.get_instance()
    
    template = """Question: {question}
Context: {context}
Initial Diagnosis: {initial_diagnosis}
Instructions: Provide a concise medical analysis following this exact format:

Diagnosis: [most relevant condition]
Symptoms: [list up to 5 key symptoms, comma-separated]
Treatment: [list up to 3 primary treatments, comma-separated]

1.  Focus on the most relevant condition based on the question, context, and initial diagnosis.
2.  List up to 5 key symptoms, separated by commas.
3.  List up to 3 primary treatments, separated by commas.
4.  Keep the entire response under 50 words.
5.  Omit all introductory text, context, and explanations.
6.  Output ONLY the formatted sections as shown above.

Answer:
"""
    
    prompt = PromptTemplate.from_template(template)
    llm_chain = prompt | llm_manager.medical_llm
    
    try:
        llm_manager.start_inference()  # Pause heartbeats during inference
        return llm_chain.invoke({
            "question": question,
            "context": context,
            "initial_diagnosis": initial_diagnosis
        })
    finally:
        llm_manager.end_inference()  # Resume heartbeats