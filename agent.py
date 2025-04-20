import streamlit as st
from groq import Groq
import PyPDF2
from docx import Document
import re
import xml.etree.ElementTree as ET

# Load Groq API key from Streamlit secrets
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

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
    if element.tag.endswith('ClinicalDocument'):
        for section in element.findall('.//section'):
            title = section.find('title')
            text = section.find('text')
            if title is not None:
                text_parts.append(title.text.strip())
            if text is not None:
                text_parts.append(text.text.strip())
    else:
        for child in element:
            if child.text and child.text.strip():
                text_parts.append(child.text.strip())
            text_parts.extend(xml_to_text(child))
    return '\n'.join(filter(None, text_parts))

def preprocess_text(text):
    """Clean and preprocess medical report text"""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'Patient ID:\s*\d+', '[REDACTED]', text)
    return text.strip()

def analyze_report(report_text):
    """Analyze medical report using Groq API"""
    system_prompt = """You are an AI medical assistant. Your role is to help users understand their medical reports by answering their questions based on the provided report text.
    Guidelines:
    
    Disclaimer: Always start your response with:"I am an AI medical assistant, not a doctor. For personalized medical advice, please consult a healthcare professional."
    
    Tone: Maintain a supportive and empathetic tone, acknowledging that medical reports can be concerning.
    
    Analysis: Analyze the report text to identify key information relevant to the user's question.  
    
    If the question is about:  
    Potential illnesses: List possible conditions mentioned or suggested by the report.  
    Critical values: Highlight any abnormal results and explain their significance.  
    Medications: Mention any prescribed or recommended medications, including generic names.  
    Lifestyle changes: Suggest any lifestyle modifications indicated in the report.  
    Follow-up tests: Note any recommended future tests or check-ups.
    
    
    For general questions, provide a summary of the report's main findings.
    
    
    Clarity: Use clear, non-technical language. Define medical terms when necessary.
    
    Urgent Concerns: If the report indicates a serious condition, urge the user to seek immediate medical attention.
    
    Limitations:  
    
    If the report text is unclear or seems incomplete, inform the user that the analysis might be limited and suggest they provide a clearer version or consult their doctor.  
    If you cannot answer the question based on the report, say:"I'm sorry, but I cannot provide an answer to that question based on the information in the report. Please consult your doctor for further assistance."  
    If you are unsure about any information, state that clearly and suggest the user verify with their doctor.
    
    
    Privacy: Do not discuss or emphasize any personal identifiers that may be present in the report.
    
    
    Your responses should be informative, accurate, and always prioritize the user's health and safety.
    """

    try:
        completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": report_text}
            ],
            temperature=0.5,
            max_completion_tokens=2048,
            top_p=0.9,
            stream=True,
        )

        st.markdown("### Medical Report Analysis")
        analysis_output = ""
        for chunk in completion:
            content = chunk.choices[0].delta.content or ""
            analysis_output += content
        st.markdown(analysis_output)
            
    except Exception as e:
        st.error(f"Error analyzing report: {e}")

def main():
    st.title("Medical Report Analyzer")
    uploaded_file = st.file_uploader("Upload a medical report", type=["pdf", "docx", "txt", "xml"])
    
    if uploaded_file is not None:
        try:
            # Save uploaded file temporarily
            with open(uploaded_file.name, "wb") as f:
                f.write(uploaded_file.getbuffer())
            raw_text = read_file(uploaded_file.name)
            cleaned_text = preprocess_text(raw_text)
            st.subheader("Sample of Extracted Text")
            st.write(cleaned_text[:200] + "...")
            analyze_report(cleaned_text)
        except Exception as e:
            st.error(f"Error processing file: {e}")

if __name__ == "__main__":
    main()
