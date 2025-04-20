HealthInsight
HealthInsight is a Streamlit-based web application designed to help users understand their medical reports. Users can upload medical reports in PDF, DOCX, TXT, or XML formats, view extracted text, receive an automated analysis, and ask specific questions about their reports. The application leverages the Groq API for natural language processing and medical report analysis.
Features

File Upload: Supports multiple file formats (PDF, DOCX, TXT, XML).
Automated Analysis: Generates a detailed analysis of the medical report, including potential illnesses, critical values, medications, lifestyle changes, and follow-up tests.
Question Interface: Allows users to ask specific questions about their reports with responses powered by the Groq API.
User-Friendly Interface: Built with Streamlit for a clean and intuitive experience.
Privacy Focus: Redacts sensitive identifiers during text preprocessing.

Prerequisites

Python 3.8+
A Groq API key (set up via environment variables)
Required Python packages (listed in requirements.txt)

Installation

Clone the Repository:
git clone https://github.com/your-username/healthinsight.git
cd healthinsight


Create a Virtual Environment (optional but recommended):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Dependencies:
pip install -r requirements.txt


Set Up Environment Variables:

Create a .env file in the project root.
Add your Groq API key:GROQ_API_KEY=your-groq-api-key


Use python-dotenv to load these variables (already included in the code).


Run the Application:
streamlit run app.py

The application will open in your default web browser at http://localhost:8501.


File Structure
healthinsight/
├── app.py              # Main Streamlit application
├── agent.py            # Backend logic for file processing and API calls
├── requirements.txt    # Python dependencies
├── .env                # Environment variables (not tracked in git)
└── README.md           # Project documentation

Usage

Upload a Medical Report:

Navigate to the application in your browser.
Use the file uploader to select a medical report (PDF, DOCX, TXT, or XML).
The extracted text will be displayed in an expandable section.


View Automated Analysis:

Once the file is processed, an automated analysis will appear, highlighting key medical insights.


Ask Questions:

Enter specific questions about the report in the text input field.
Responses will be displayed in a conversation history section.


Note:

Always consult a healthcare professional for personalized medical advice. HealthInsight is an informational tool, not a substitute for professional medical guidance.



Dependencies
Listed in requirements.txt:
streamlit
groq
PyPDF2
python-docx
python-dotenv

Additional dependencies may be required based on your environment (e.g., lxml for XML processing).
System Prompt
The application uses a carefully crafted system prompt to ensure effective and safe responses:

Emphasizes that the AI is not a doctor and professional consultation is necessary.
Provides clear, non-technical explanations.
Highlights urgent concerns and suggests follow-up actions.
Maintains privacy by avoiding discussion of personal identifiers.

Limitations

The application relies on the quality and clarity of the uploaded report. Incomplete or poorly formatted reports may result in limited analysis.
The Groq API has usage limits; ensure your API key has sufficient quota.
Currently supports only PDF, DOCX, TXT, and XML formats.

Contributing
Contributions are welcome! To contribute:

Fork the repository.
Create a new branch (git checkout -b feature/your-feature).
Commit your changes (git commit -m 'Add your feature').
Push to the branch (git push origin feature/your-feature).
Open a pull request.

License
This project is licensed under the MIT License. See the LICENSE file for details.
Contact
For questions or support, please open an issue on the GitHub repository or contact the maintainer at [your-email@example.com].

Disclaimer: HealthInsight is an informational tool and not a substitute for professional medical advice. Always consult a healthcare professional for medical decisions.
