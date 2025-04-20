import streamlit as st
from agent import read_file, preprocess_text, analyze_report
from groq import Groq
import os
import uuid

# Initialize Groq client with API key from Streamlit secrets
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# Streamlit page configuration
st.set_page_config(page_title="HealthInsight", page_icon="🏥", layout="wide")

# Session state initialization
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'report_text' not in st.session_state:
    st.session_state.report_text = ""
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def ask_question(question, report_text):
    """Ask a question about the medical report using Groq API"""
    system_prompt = """
    You are an AI medical assistant. Your role is to help users understand their medical reports by answering their questions based on the provided report text.
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
                {"role": "user", "content": f"Report: {report_text}\n\nQuestion: {question}"}
            ],
            temperature=0.5,
            max_completion_tokens=1024,
            top_p=0.9,
            stream=False,
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error processing question: {str(e)}"

def main():
    st.title("HealthInsight")
    st.markdown("Upload your medical report and ask questions about it. Supports PDF, DOCX, TXT, and XML formats.")

    # File upload section
    st.subheader("Upload Medical Report")
    uploaded_file = st.file_uploader(
        "Choose a medical report file",
        type=['pdf', 'docx', 'txt', 'xml'],
        key="file_uploader"
    )

    if uploaded_file is not None:
        # Save uploaded file temporarily
        file_extension = uploaded_file.name.split('.')[-1]
        temp_file_path = f"temp_{uuid.uuid4()}.{file_extension}"
        
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        try:
            # Read and preprocess the file
            raw_text = read_file(temp_file_path)
            st.session_state.report_text = preprocess_text(raw_text)
            st.session_state.uploaded_file = uploaded_file.name
            
            # Display extracted text
            with st.expander("View Extracted Report Text"):
                st.text_area(
                    "Extracted Text",
                    st.session_state.report_text,
                    height=200,
                    disabled=True
                )
            
            # Display automated analysis
            st.subheader("Automated Report Analysis")
            with st.spinner("Analyzing report..."):
                analysis_placeholder = st.empty()
                # Since analyze_report prints directly, we'll capture output differently
                import sys
                from io import StringIO
                
                old_stdout = sys.stdout
                sys.stdout = mystdout = StringIO()
                analyze_report(st.session_state.report_text)
                sys.stdout = old_stdout
                analysis_placeholder.markdown(mystdout.getvalue())
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
    
    # Question input section
    if st.session_state.report_text:
        st.subheader("Ask Questions About Your Report")
        question = st.text_input("Enter your question:")
        
        if question:
            with st.spinner("Processing your question..."):
                answer = ask_question(question, st.session_state.report_text)
                st.session_state.chat_history.append({"question": question, "answer": answer})
        
        # Display chat history
        if st.session_state.chat_history:
            st.subheader("Conversation History")
            for chat in st.session_state.chat_history:
                st.markdown(f"**Q: {chat['question']}**")
                st.markdown(f"A: {chat['answer']}")
                st.markdown("---")

if __name__ == "__main__":
    main()