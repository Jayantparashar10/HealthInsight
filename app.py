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
    system_prompt = """You are a medical expert AI assistant. Answer questions about the provided medical report concisely and accurately. Always advise consulting a healthcare professional for final decisions."""
    
    try:
        completion = client.chat.completions.create(
            model="mixtral-8x7b-32768",
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