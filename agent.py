from groq import Groq
import PyPDF2
from docx import Document
import re
import xml.etree.ElementTree as ET
from dotenv import load_dotenv
load_dotenv()

client = Groq()

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
    system_prompt = """You are a medical expert AI assistant. Analyze the provided medical report and:
    1. Identify potential illnesses or health issues
    2. Highlight critical values that need immediate attention
    3. Suggest recommended medications (include generic names)
    4. Recommend lifestyle changes
    5. Mention necessary follow-up tests
    6. Always advise consulting a healthcare professional

    Present results in clear sections with markdown formatting. Use conservative medical judgment."""

    try:
        completion = client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": report_text}
            ],
            temperature=0.5,
            max_completion_tokens=2048,
            top_p=0.9,
            stream=True,
        )

        print("\nMedical Report Analysis:\n")
        for chunk in completion:
            print(chunk.choices[0].delta.content or "", end="")
            
    except Exception as e:
        print(f"Error analyzing report: {e}")

def main():
    file_path = "1.xml"  # Change to your file path
    
    try:
        print(f"Reading medical report from: {file_path}")
        raw_text = read_file(file_path)
        cleaned_text = preprocess_text(raw_text)
        
        # Print first 200 chars for verification
        print("\nSample of extracted text:")
        print(cleaned_text[:200] + "...")
        
        analyze_report(cleaned_text)
        
    except Exception as e:
        print(f"Error processing file: {e}")

if __name__ == "__main__":
    main()