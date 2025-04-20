import streamlit as st
from groq import Groq
import os
import uuid
import io
import base64
from PIL import Image
import PyPDF2
from docx import Document
import re
import xml.etree.ElementTree as ET

# Initialize Groq client with API key from Streamlit secrets
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

def github_model_chat_completion(messages, temperature=0.5, max_tokens=1024, top_p=0.9):
    url = "https://api.github.com/models/chat/completions"
    headers = {
        "Authorization": f"Bearer {st.secrets['GITHUB_MODELS_TOKEN']}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    payload = {
        "model": "azureml-meta/Llama-3-2-90B-Vision-Instruct",
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p
    }
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]
# Streamlit page configuration
st.set_page_config(page_title="HealthInsight", page_icon="ðŸ¥", layout="wide")

# Session state initialization
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'report_text' not in st.session_state:
    st.session_state.report_text = None
if 'uploaded_file_name' not in st.session_state:
    st.session_state.uploaded_file_name = None
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None

def read_file(file_path):
    """Read medical report from different file formats"""
    if file_path.endswith('.pdf'):
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ''.join([page.extract_text() for page in reader.pages])
            return text
        
    elif file_path.endswith('.docx'):
        doc = Document(file_path)
        return '\n'.join([para.text for para in doc.paragraphs])
    
    elif file_path.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    elif file_path.endswith('.xml'):
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            return xml_to_text(root)
        except ET.ParseError as e:
            raise ValueError(f"Invalid XML file: {str(e)}")
    
    else:
        raise ValueError("Unsupported file format. Use PDF, DOCX, TXT, or XML")

def xml_to_text(element):
    """Convert XML elements to readable text"""
    text_parts = []
    
    # Handle common medical XML structures
    if element.tag.endswith('ClinicalDocument'):
        # Handle HL7 CDA format
        for section in element.findall('.//section'):
            title = section.find('title')
            text = section.find('text')
            if title is not None:
                text_parts.append(title.text.strip())
            if text is not None:
                text_parts.append(text.text.strip())
    else:
        # Generic XML handling
        for child in element:
            if child.text and child.text.strip():
                text_parts.append(child.text.strip())
            text_parts.extend(xml_to_text(child))
    
    return '\n'.join(filter(None, text_parts))

def preprocess_text(text):
    """Clean and preprocess medical report text"""
    # Remove multiple spaces and newlines
    text = re.sub(r'\s+', ' ', text)
    # Remove sensitive identifiers (basic example)
    text = re.sub(r'Patient ID:\s*\d+', '[REDACTED]', text)
    return text.strip()

def analyze_report(report_text):
    """Analyze medical report using Groq API"""
    # Using the new system prompt for consistency
    system_prompt = """
You are a doctor. Your role is to help users understand their medical reports by answering their questions based on the provided report text.
Guidelines:

Tone: Maintain a supportive and empathetic tone, acknowledging that medical reports can be concerning.

Analysis: Analyze the report text to identify key information relevant to the user's question.  

If the question is about or indicates:  
Potential illnesses: List possible conditions mentioned or suggested by the report.  
Critical values: Highlight any abnormal results and explain their significance.  
Medications: Suggest recommended medications, including generic names, based on the report's findings.  
Home Remedies: Provide steps for home remedies where applicable and safe, emphasizing they are supplementary and not a substitute for professional care.  
Follow-up Tests: Recommend necessary follow-up tests or diagnostics based on the condition.  
Severe Conditions: If the condition appears serious or life-threatening, suggest urgent medical attention, additional specialist consultations, and any critical tests or interventions that might be needed.


For general questions, provide a summary of the report's main findings.


Clarity: Use clear, non-technical language. Define medical terms when necessary.

Urgent Concerns: If the report indicates a serious condition (e.g., heart attack, cancer, severe infection), urge the user to seek immediate medical attention and suggest emergency steps if applicable.

Limitations:  

If the report text is unclear or seems incomplete, inform the user that the analysis might be limited and suggest they provide a clearer version or consult their doctor.  
If you cannot answer the question based on the report, say:"I'm sorry, but I cannot provide an answer to that question based on the information in the report. Please consult your doctor for further assistance."  


Privacy: Do not discuss or emphasize any personal identifiers that may be present in the report.


Your responses should be informative, accurate, and always prioritize the user's health and safety.
"""

    try:
        return github_model_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Please analyze this medical report and provide a comprehensive summary: " + report_text}
            ],
            temperature=0.5,
            max_tokens=1024,
            top_p=0.9
        )
        
    except Exception as e:
        return f"Error analyzing report: {e}"

def process_image(image):
    """Process and analyze medical image using Groq API"""
    # Convert the image to a base64 string
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    # Using the new system prompt for consistency
    system_prompt = """System Prompt for Medical Image Analysis
You are a doctor specialized in analyzing medical images (e.g., X-rays, MRIs, CT scans, ultrasounds). Your role is to provide expert insights based on the visual data from the uploaded medical images.
Guidelines:


Tone: Maintain a professional, supportive, and empathetic tone, acknowledging that medical imaging results can be concerning.

Analysis: Analyze the provided medical image to identify key visual findings relevant to the user's query or the image's context.  

If the image suggests:  
Potential conditions: Identify possible abnormalities or diseases (e.g., fractures, tumors, infections) based on visible patterns or structures.  
Critical findings: Highlight any urgent or abnormal features (e.g., signs of bleeding, organ enlargement) and explain their potential significance.  
Medications: Suggest recommended medications (including generic names) if a condition is identifiable and treatment is implied, noting these are preliminary suggestions.  
Home Remedies: Provide steps for home remedies where applicable and safe (e.g., rest for minor injuries), emphasizing they are supplementary and not a substitute for professional care.  
Follow-up Tests: Recommend additional imaging or diagnostic tests (e.g., MRI for unclear X-ray findings) to confirm or expand on the analysis.  
Severe Conditions: If the image indicates a serious or life-threatening condition (e.g., massive stroke, advanced cancer), urge the user to seek immediate medical attention, suggest specialist referrals, and recommend critical tests or interventions.


For general queries, provide a summary of observed findings and their potential implications.


Clarity: Use clear, non-technical language. Define medical imaging terms (e.g., "opacity" or "lesion") when necessary.

Urgent Concerns: If the image shows signs of a serious condition (e.g., acute hemorrhage, large mass), urge the user to seek immediate medical attention and suggest emergency steps if applicable.

Limitations:  

If the image quality is poor, incomplete, or unclear, inform the user that the analysis may be limited and suggest they provide a higher-quality image or consult a radiologist.  
If you cannot identify a condition or answer the question based on the image, say:"I'm sorry, but I cannot provide a definitive analysis or answer based on this image. Please consult a radiologist or doctor for further evaluation."  
If you are unsure about any findings (e.g., rare conditions, treatment options), state that clearly and suggest the user verify with a medical professional.


Privacy: Do not discuss or emphasize any personal identifiers that may be present in the image or associated data.


Your responses should be informative, accurate, and always prioritize the user's health and safety. Provide your analysis based solely on the visual content of the medical image, avoiding speculation beyond what is visually evident.
"""

    try:
        # Note: For image analysis, we need a model with vision capabilities
        # If Groq doesn't support this directly, you might need to use a different provider
        # or extract features from the image first
         return github_model_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Please analyze this medical report and provide a comprehensive summary: " + report_text}
            ],
            temperature=0.5,
            max_tokens=1024,
            top_p=0.9
        )
    except Exception as e:
        return f"Error analyzing image: {e}"

def chat_with_context(message, report_text=None, image=None):
    """Generate a response based on the message and any medical context"""
    # Updated system prompt as requested
    system_prompt = """
You are a doctor. Your role is to help users understand their medical reports by answering their questions based on the provided report text.
Guidelines:

Tone: Maintain a supportive and empathetic tone, acknowledging that medical reports can be concerning.

Analysis: Analyze the report text to identify key information relevant to the user's question.  

If the question is about or indicates:  
Potential illnesses: List possible conditions mentioned or suggested by the report.  
Critical values: Highlight any abnormal results and explain their significance.  
Medications: Suggest recommended medications, including generic names, based on the report's findings.  
Home Remedies: Provide steps for home remedies where applicable and safe, emphasizing they are supplementary and not a substitute for professional care.  
Follow-up Tests: Recommend necessary follow-up tests or diagnostics based on the condition.  
Severe Conditions: If the condition appears serious or life-threatening, suggest urgent medical attention, additional specialist consultations, and any critical tests or interventions that might be needed.


For general questions, provide a summary of the report's main findings.


Clarity: Use clear, non-technical language. Define medical terms when necessary.

Urgent Concerns: If the report indicates a serious condition (e.g., heart attack, cancer, severe infection), urge the user to seek immediate medical attention and suggest emergency steps if applicable.

Limitations:  

If the report text is unclear or seems incomplete, inform the user that the analysis might be limited and suggest they provide a clearer version or consult their doctor.  
If you cannot answer the question based on the report, say:"I'm sorry, but I cannot provide an answer to that question based on the information in the report. Please consult your doctor for further assistance."  


Privacy: Do not discuss or emphasize any personal identifiers that may be present in the report.


Your responses should be informative, accurate, and always prioritize the user's health.
"""
    
    context = "User query: " + message
    
    if report_text:
        context += "\n\nMedical report content: " + report_text
        
    if image:
        context += "\n\nNote: User has also uploaded a medical image."
    
    try:
         return github_model_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Please analyze this medical report and provide a comprehensive summary: " + report_text}
            ],
            temperature=0.5,
            max_tokens=1024,
            top_p=0.9
        )
        
    except Exception as e:
        return f"Error generating response: {e}"

def save_uploaded_file(uploaded_file):
    """Save an uploaded file temporarily and return the path"""
    file_extension = uploaded_file.name.split('.')[-1].lower()
    temp_file_path = f"temp_{uuid.uuid4()}.{file_extension}"
    
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return temp_file_path

def main():
    # App title and description
    st.title("HealthInsight")
    st.markdown("Chat with or without medical reports and images. Get insights about your health information.")
    
    # Sidebar for file uploads and settings
    with st.sidebar:
        st.header("Upload Medical Information")
        
        # Medical report upload
        report_tab, image_tab = st.tabs(["Medical Report", "Medical Image"])
        
        with report_tab:
            report_file = st.file_uploader(
                "Upload a medical report",
                type=['pdf', 'docx', 'txt', 'xml'],
                key="report_uploader"
            )
            
            if report_file:
                temp_file_path = save_uploaded_file(report_file)
                
                try:
                    # Process the file
                    raw_text = read_file(temp_file_path)
                    st.session_state.report_text = preprocess_text(raw_text)
                    st.session_state.uploaded_file_name = report_file.name
                    
                    st.success(f"âœ… Report loaded: {report_file.name}")
                    
                    # Show preview
                    with st.expander("Report Preview"):
                        preview_text = st.session_state.report_text[:300] + "..." if len(st.session_state.report_text) > 300 else st.session_state.report_text
                        st.text_area("Content", preview_text, height=150, disabled=True)
                    
                    # Analyze button
                    if st.button("Analyze Report"):
                        with st.spinner("Analyzing report..."):
                            analysis = analyze_report(st.session_state.report_text)
                            # Add system message to chat
                            st.session_state.chat_history.append({
                                "role": "assistant", 
                                "content": f"ðŸ“‹ **Report Analysis**\n\n{analysis}"
                            })
                            st.success("Analysis complete! Check the chat area.")
                
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                finally:
                    # Clean up
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)
        
        with image_tab:
            image_file = st.file_uploader(
                "Upload a medical image",
                type=['png', 'jpg', 'jpeg'],
                key="image_uploader"
            )
            
            if image_file:
                try:
                    image = Image.open(image_file)
                    st.session_state.uploaded_image = image
                    st.image(image, caption="Uploaded image", use_column_width=True)
                    
                    # Analyze button
                    if st.button("Analyze Image"):
                        with st.spinner("Analyzing image..."):
                            analysis = process_image(image)
                            # Add system message to chat
                            st.session_state.chat_history.append({
                                "role": "assistant", 
                                "content": f"ðŸ–¼ï¸ **Image Analysis**\n\n{analysis}"
                            })
                            st.success("Analysis complete! Check the chat area.")
                
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        # Clear button
        if st.button("Clear All Uploads"):
            st.session_state.report_text = None
            st.session_state.uploaded_file_name = None
            st.session_state.uploaded_image = None
            st.success("All uploads cleared!")
    
    # Main chat interface
    st.header("Chat")
    
    # Display chat messages
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    prompt = "Ask about your health or uploaded medical information..."
    user_message = st.chat_input(prompt)
    
    if user_message:
        # Add user message to chat
        with st.chat_message("user"):
            st.markdown(user_message)
        st.session_state.chat_history.append({"role": "user", "content": user_message})
        
        # Generate response
        with st.spinner("Generating response..."):
            response = chat_with_context(
                user_message,
                report_text=st.session_state.report_text,
                image=st.session_state.uploaded_image
            )
        
        # Display response
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.chat_history.append({"role": "assistant", "content": response})
    
    # Option to clear chat history
    if st.session_state.chat_history and st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.success("Chat history cleared!")

if __name__ == "__main__":
    main()
