from langchain_core.prompts import PromptTemplate
import base64
from singleton import LLMManager

def image_to_base64_data_uri(file_path):
    with open(file_path, "rb") as img_file:
        base64_data = base64.b64encode(img_file.read()).decode('utf-8')
        return f"data:image/png;base64,{base64_data}"

def init_llm_input(question, image=None):
    """Process input with multimodal LLM"""
    llm_manager = LLMManager.get_instance()
    
    if image is None:
        template = """Question: {question}
ATTENTION: THIS IS A LIST-ONLY RESPONSE.
Instructions:
1. List ONLY disease names (upto 5)
2. Separate with commas
3. NO explanations
4. NO additional text
5. NO symptoms
6. NO details

Answer: """
    else:
        template = """Question: {question} and image data: {image}
ATTENTION: THIS IS A LIST-ONLY RESPONSE.
Instructions:
1. List ONLY disease names
2. Separate with commas
3. NO explanations
4. NO additional text
5. NO symptoms
6. NO details

Answer: """
    
    prompt = PromptTemplate.from_template(template)
    llm_chain = prompt | llm_manager.llm
    
    return llm_chain.invoke({
        "question": question,
        "image": image
    })

def post_llm_input(initial_diagnosis,question, context):
    """Process follow-up with context"""
    llm_manager = LLMManager.get_instance()
    
    template = """Question: {question}
    Context: {context}
    Initial Diagnosis: {initial_diagnosis}
Instructions: Provide a concise medical analysis following this exact format:

Diagnosis: [single most relevant condition]
Symptoms: [list up to 5 key symptoms]
Treatment: [list up to 3 primary treatments]

Keep response under 50 words. Omit all introductory text, context, and explanations.
Output the formatted sections only.

Answer:
 """
    
    prompt = PromptTemplate.from_template(template)
    llm_chain = prompt | llm_manager.medical_llm
    return llm_chain.invoke({
        "question": question,
        "context": context,
        "initial_diagnosis": initial_diagnosis
    })