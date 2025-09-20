# -*- coding: utf-8 -*-
import streamlit as st
import PyPDF2
import numpy as np
import re
import tempfile
import os
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Set up the page - MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="StudyMate Pro - AI-Powered PDF Learning Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-card {
        background-color: #E3F2FD;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .success-box {
        background-color: #E8F5E9;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .answer-box {
        background-color: #FFF3E0;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .stButton>button {
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
    }
    .flashcard {
        background-color: #E3F2FD;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #1E88E5;
    }
    .timer-container {
        text-align: center;
        padding: 20px;
        background-color: #f0f0f0;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'pdf_text' not in st.session_state:
    st.session_state.pdf_text = ""
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'chunks' not in st.session_state:
    st.session_state.chunks = []
if 'study_data' not in st.session_state:
    st.session_state.study_data = {
        'sessions': [],
        'pages_read': 0,
        'flashcards_created': 0,
        'quizzes_taken': 0
    }
if 'current_timer' not in st.session_state:
    st.session_state.current_timer = None

# Utility functions
def extract_text_from_pdf(uploaded_file):
    """Extract text from a PDF file"""
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text, len(pdf_reader.pages)

def preprocess_text(text):
    """Clean and chunk the extracted text"""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Split into chunks
    chunks = []
    current_chunk = ""
    sentences = text.split('. ')
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < 500:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk)
            current_chunk = sentence + ". "
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

def generate_content(prompt, context_chunks, max_length=500):
    """Generate content based on prompt and context"""
    context = "\n".join(context_chunks[:3])  # Use first 3 chunks
    
    # Simulate different types of responses
    if "summary" in prompt.lower():
        return f"Summary:\n\n{context[:max_length]}...\n\n[This is a simulated summary. In production, this would be AI-generated.]"
    elif "flashcard" in prompt.lower():
        return f"Flashcard:\n\nQ: What is the main topic?\nA: {context[:200]}...\n\n[Simulated flashcard content]"
    elif "quiz" in prompt.lower():
        return "Quiz Question:\n\nWhat is the main concept?\nA) Option 1\nB) Option 2\nC) Option 3\nD) Option 4\n\nCorrect answer: B\n\n[Simulated quiz question]"
    elif "keyword" in prompt.lower():
        keywords = list(set(re.findall(r'\b[A-Z][a-z]+\b', context)))[:5]
        return f"Key Terms: {', '.join(keywords)}"
    else:
        return f"Answer:\n\nBased on the provided materials: {context[:max_length]}...\n\n[Simulated AI response]"

# Header section
st.markdown('<h1 class="main-header">StudyMate Pro</h1>', unsafe_allow_html=True)
st.markdown("### Your AI-Powered Academic Assistant with Advanced Features")
st.markdown("---")

# Sidebar for navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose a feature", 
                               ["Home", "PDF Q&A", "Smart Summarizer", "Flashcards Generator", 
                                "Quiz Maker", "Study Timer"])

# Home page
if app_mode == "Home":
    st.markdown("StudyMate Pro is an advanced AI-powered academic assistant that enables students to interact with their study materials in multiple ways. Upload your PDFs and access a suite of learning tools designed to enhance your study experience.")
    
    # Features overview
    st.markdown("### Key Features")
    cols = st.columns(3)
    
    features = [
        ("PDF Q&A", "Ask questions about your documents and get AI-powered answers"),
        ("Smart Summarizer", "Generate concise summaries of lengthy documents"),
        ("Flashcards", "Create study flashcards from your materials automatically"),
        ("Quiz Maker", "Generate practice quizzes to test your knowledge"),
        ("Study Timer", "Pomodoro timer and progress tracking"),
        ("Progress Tracking", "Monitor your study activities and progress")
    ]
    
    for i, (title, desc) in enumerate(features):
        with cols[i % 3]:
            st.markdown(f'<div class="feature-card"><h4>{title}</h4><p>{desc}</p></div>', unsafe_allow_html=True)

# PDF Processing (common to many features)
if app_mode != "Home":
    uploaded_files = st.file_uploader(
        "Upload your PDF files", 
        type=['pdf'], 
        accept_multiple_files=True,
        help="You can upload one or multiple PDF files"
    )
    
    if uploaded_files and not st.session_state.processed:
        if st.button("Process PDFs"):
            with st.spinner("Processing your PDFs..."):
                all_text = ""
                total_pages = 0
                for uploaded_file in uploaded_files:
                    text, pages = extract_text_from_pdf(uploaded_file)
                    all_text += text + "\n\n"
                    total_pages += pages
                
                st.session_state.pdf_text = all_text
                st.session_state.chunks = preprocess_text(all_text)
                st.session_state.processed = True
                st.session_state.total_pages = total_pages
                
                # Update study data
                st.session_state.study_data['pages_read'] += total_pages
                st.session_state.study_data['sessions'].append({
                    'date': datetime.now().isoformat(),
                    'pages': total_pages,
                    'action': 'processed'
                })
                
                st.success(f"Processed {len(uploaded_files)} PDF(s) with {total_pages} pages and extracted {len(st.session_state.chunks)} text chunks!")

# PDF Q&A Feature
if app_mode == "PDF Q&A" and st.session_state.processed:
    st.markdown("### Ask Questions About Your Documents")
    
    question = st.text_input(
        "Enter your question:",
        placeholder="e.g., What are the main concepts discussed in chapter 3?",
        help="Ask any question about the content of your uploaded PDFs"
    )
    
    difficulty = st.select_slider("Explanation Level", 
                                 options=["Simple", "Intermediate", "Advanced"],
                                 value="Intermediate")
    
    if question:
        with st.spinner("Finding relevant information..."):
            # Generate answer based on difficulty
            prompt = f"Provide a {difficulty.lower()} level explanation: {question}"
            answer = generate_content(prompt, st.session_state.chunks)
            
            # Display answer
            st.markdown("### Answer")
            st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

# Smart Summarizer
elif app_mode == "Smart Summarizer" and st.session_state.processed:
    st.markdown("### Generate Smart Summaries")
    
    summary_length = st.select_slider("Summary Length", 
                                     options=["Very Short", "Short", "Medium", "Detailed"],
                                     value="Medium")
    
    if st.button("Generate Summary"):
        with st.spinner("Creating your summary..."):
            # Generate summary
            prompt = f"Create a {summary_length.lower()} summary of the key points"
            summary = generate_content(prompt, st.session_state.chunks, max_length=1000)
            
            st.markdown("### Document Summary")
            st.markdown(f'<div class="answer-box">{summary}</div>', unsafe_allow_html=True)
            
            # Download option
            st.download_button("Download Summary", summary, file_name="document_summary.txt")

# Flashcards Generator
elif app_mode == "Flashcards Generator" and st.session_state.processed:
    st.markdown("### Generate Study Flashcards")
    
    num_flashcards = st.slider("Number of Flashcards", 3, 10, 5)
    
    if st.button("Generate Flashcards"):
        with st.spinner("Creating your flashcards..."):
            # Generate flashcards
            flashcards = []
            for i in range(num_flashcards):
                prompt = f"Create flashcard {i+1} with a question and answer"
                flashcard_content = generate_content(prompt, st.session_state.chunks)
                
                # Parse the flashcard content
                if "Q:" in flashcard_content and "A:" in flashcard_content:
                    parts = flashcard_content.split("A:")
                    question = parts[0].replace("Q:", "").strip()
                    answer = parts[1].strip()
                else:
                    question = f"Question {i+1} about the material"
                    answer = f"Answer {i+1} based on the content"
                
                flashcards.append({"question": question, "answer": answer})
            
            st.session_state.flashcards = flashcards
            st.session_state.study_data['flashcards_created'] += len(flashcards)
        
        st.success(f"Generated {num_flashcards} flashcards!")
        
        # Display flashcards
        for i, card in enumerate(flashcards):
            st.markdown(f"""
            <div class="flashcard">
                <h3>Q: {card['question']}</h3>
                <p><strong>A:</strong> {card['answer']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Download option
        flashcard_text = "\n\n".join([f"Q: {card['question']}\nA: {card['answer']}" for card in flashcards])
        st.download_button("Download Flashcards", flashcard_text, file_name="flashcards.txt")

# Quiz Maker
elif app_mode == "Quiz Maker" and st.session_state.processed:
    st.markdown("### Generate Practice Quiz")
    
    quiz_type = st.radio("Quiz Type", ("Multiple Choice", "True/False"))
    num_questions = st.slider("Number of Questions", 3, 10, 5)
    
    if st.button("Generate Quiz"):
        with st.spinner("Creating your quiz..."):
            # Generate quiz questions
            quiz_questions = []
            for i in range(num_questions):
                prompt = f"Create a {quiz_type.lower()} question {i+1} with options and correct answer"
                question_content = generate_content(prompt, st.session_state.chunks)
                quiz_questions.append(question_content)
            
            st.session_state.quiz_questions = quiz_questions
            st.session_state.study_data['quizzes_taken'] += 1
        
        st.success(f"Generated {num_questions} {quiz_type} questions!")
        
        # Display quiz
        for i, question in enumerate(quiz_questions):
            st.markdown(f"Question {i+1}:")
            st.write(question)
            st.text_input(f"Your answer for question {i+1}", key=f"quiz_ans_{i}")
            st.markdown("---")
        
        if st.button("Check Answers"):
            st.info("In a full implementation, this would validate your answers against the correct ones.")

# Study Timer
elif app_mode == "Study Timer":
    st.markdown("### Study Timer & Progress Tracker")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Pomodoro Timer")
        study_time = st.number_input("Study minutes", min_value=5, max_value=60, value=25)
        break_time = st.number_input("Break minutes", min_value=5, max_value=30, value=5)
        
        if st.button("Start Timer"):
            st.session_state.current_timer = {
                'start_time': datetime.now(),
                'end_time': datetime.now() + timedelta(minutes=study_time),
                'type': 'study'
            }
            st.success(f"Study timer started for {study_time} minutes!")
    
    with col2:
        st.markdown("#### Progress Statistics")
        st.metric("Pages Read", st.session_state.study_data.get('pages_read', 0))
        st.metric("Flashcards Created", st.session_state.study_data.get('flashcards_created', 0))
        st.metric("Quizzes Taken", st.session_state.study_data.get('quizzes_taken', 0))
        
        if st.session_state.study_data.get('sessions'):
            last_session = st.session_state.study_data['sessions'][-1]
            st.write(f"Last study session: {datetime.fromisoformat(last_session['date']).strftime('%Y-%m-%d %H:%M')}")
    
    # Timer display
    if st.session_state.current_timer:
        timer = st.session_state.current_timer
        time_left = timer['end_time'] - datetime.now()
        
        if time_left.total_seconds() > 0:
            minutes, seconds = divmod(int(time_left.total_seconds()), 60)
            st.markdown(f"""
            <div class="timer-container">
                <h2>{minutes:02d}:{seconds:02d}</h2>
                <p>{'Study' if timer['type'] == 'study' else 'Break'} time remaining</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.session_state.current_timer = None

# Footer section
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>StudyMate Pro uses advanced AI technologies including:</p>
        <p>Python, Streamlit, PyPDF2, and simulated AI responses</p>
    </div>
""", unsafe_allow_html=True)
