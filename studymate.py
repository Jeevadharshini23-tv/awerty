import warnings
warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")

import streamlit as st
import PyPDF2
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import re
import tempfile
import os
import json
from datetime import datetime, timedelta
import time
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
import base64

# Try to import the voice-related packages with error handling
try:
    from gtts import gTTS
    import io
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    st.warning("Text-to-speech functionality is not available. Please install gTTS with: pip install gTTS")

# Try to import translation with error handling
try:
    from googletrans import Translator
    TRANSLATION_AVAILABLE = True
except ImportError:
    TRANSLATION_AVAILABLE = False
    st.warning("Translation functionality is not available. Please install googletrans with: pip install googletrans==4.0.0rc1")

import random

# Set up the page
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
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-bottom: 1rem;
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
    .tab-content {
        padding: 1rem;
    }
    .highlight {
        background-color: yellow;
        font-weight: bold;
    }
    .flashcard {
        perspective: 1000px;
        width: 300px;
        height: 200px;
        margin: 10px;
    }
    .flashcard-inner {
        position: relative;
        width: 100%;
        height: 100%;
        text-align: center;
        transition: transform 0.6s;
        transform-style: preserve-3d;
    }
    .flashcard:hover .flashcard-inner {
        transform: rotateY(180deg);
    }
    .flashcard-front, .flashcard-back {
        position: absolute;
        width: 100%;
        height: 100%;
        backface-visibility: hidden;
        display: flex;
        align-items: center;
        justify-content: center;
        border: 1px solid #ccc;
        border-radius: 10px;
        padding: 20px;
    }
    .flashcard-front {
        background-color: #1E88E5;
        color: white;
    }
    .flashcard-back {
        background-color: #f5f5f5;
        color: #333;
        transform: rotateY(180deg);
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
if 'faiss_index' not in st.session_state:
    st.session_state.faiss_index = None
if 'chunks' not in st.session_state:
    st.session_state.chunks = []
if 'model' not in st.session_state:
    st.session_state.model = None
if 'study_data' not in st.session_state:
    st.session_state.study_data = {
        'sessions': [],
        'pages_read': 0,
        'flashcards_created': 0,
        'quizzes_taken': 0
    }
if 'current_timer' not in st.session_state:
    st.session_state.current_timer = None
if 'collaboration_mode' not in st.session_state:
    st.session_state.collaboration_mode = False
if 'shared_notes' not in st.session_state:
    st.session_state.shared_notes = []

# Initialize models
@st.cache_resource
def load_models():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    translator = Translator() if TRANSLATION_AVAILABLE else None
    return model, translator

model, translator = load_models()

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
    
    # Split into chunks (simplified approach)
    chunks = []
    current_chunk = ""
    sentences = text.split('. ')
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < 500:  # Rough chunk size limit
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk)
            current_chunk = sentence + ". "
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

def setup_faiss_index(chunks):
    """Create a FAISS index for semantic search"""
    embeddings = model.encode(chunks)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index, embeddings

def find_relevant_chunks(question, chunks, index, embeddings, k=3):
    """Find the most relevant text chunks for a question"""
    question_embedding = model.encode([question])
    distances, indices = index.search(question_embedding, k)
    
    relevant_chunks = [chunks[i] for i in indices[0]]
    return relevant_chunks

def generate_content(prompt, context_chunks, max_length=500):
    """Generate content based on prompt and context"""
    context = "\n".join(context_chunks)
    
    full_prompt = f"""
    Based on the following context from academic materials:
    
    {context}
    
    {prompt}
    """
    
    # In a real implementation, you would use IBM Watsonx or another LLM here
    # For demonstration, we'll simulate different types of responses
    
    if "summary" in prompt.lower():
        return f"Summary of the content: {context[:max_length]}... [This is a simulated summary. In a full implementation, this would be generated by an advanced LLM]"
    elif "flashcard" in prompt.lower():
        return f"Q: What is the main topic?\nA: {context[:200]}... [Simulated flashcard]"
    elif "quiz" in prompt.lower():
        return "Q: What is the main concept?\nA) Option 1\nB) Option 2\nC) Option 3\nD) Option 4\nCorrect answer: B) Option 2\n[Simulated quiz question]"
    elif "keyword" in prompt.lower():
        keywords = list(set(re.findall(r'\b[A-Z][a-z]+\b', context)))[:5]
        return f"Key terms: {', '.join(keywords)}"
    else:
        return f"Based on the provided materials: {context[:max_length]}... [Simulated response]"

def text_to_speech(text, lang='en'):
    """Convert text to speech"""
    if not TTS_AVAILABLE:
        return None
        
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        return audio_bytes
    except Exception as e:
        st.error(f"Text-to-speech error: {e}")
        return None

def translate_text(text, dest_language):
    """Translate text to another language"""
    if not TRANSLATION_AVAILABLE:
        return f"Translation to {dest_language} not available. Please install googletrans."
    
    try:
        translation = translator.translate(text, dest=dest_language)
        return translation.text
    except Exception as e:
        return f"Translation error: {e}"

def generate_mind_map(chunks):
    """Generate a simple mind map from text chunks"""
    # Extract key concepts (simplified)
    key_concepts = []
    for chunk in chunks[:5]:  # Use first 5 chunks for demo
        words = re.findall(r'\b[A-Z][a-z]+\b', chunk)
        if words:
            key_concepts.extend(words[:3])
    
    # Create a simple graph
    G = nx.Graph()
    main_topic = "Study Material"
    G.add_node(main_topic)
    
    for i, concept in enumerate(set(key_concepts[:6])):
        G.add_node(concept)
        G.add_edge(main_topic, concept)
        
        # Add some sub-concepts for demo
        for j in range(2):
            sub_concept = f"{concept}_{j+1}"
            G.add_node(sub_concept)
            G.add_edge(concept, sub_concept)
    
    return G

# Header section
st.markdown('<h1 class="main-header">StudyMate Pro</h1>', unsafe_allow_html=True)
st.markdown("### Your AI-Powered Academic Assistant with Advanced Features")
st.markdown("---")

# Sidebar for navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose a feature", 
                               ["Home", "PDF Q&A", "Smart Summarizer", "Flashcards Generator", 
                                "Quiz Maker", "Keyword Highlighter", "Voice Assistant", 
                                "Multilingual Support", "Mind Map Generator", "Study Timer"])

# Home page
if app_mode == "Home":
    st.markdown("""
        StudyMate Pro is an advanced AI-powered academic assistant that enables students to interact with their study materials 
        in multiple ways. Upload your PDFs and access a suite of learning tools designed to enhance your study experience.
    """)
    
    # Features overview
    st.markdown("### Key Features")
    cols = st.columns(3)
    
    features = [
        ("📚 PDF Q&A", "Ask questions about your documents and get AI-powered answers"),
        ("📝 Smart Summarizer", "Generate concise summaries of lengthy documents"),
        ("🔖 Flashcards", "Create study flashcards from your materials automatically"),
        ("❓ Quiz Maker", "Generate practice quizzes to test your knowledge"),
        ("🔍 Keyword Highlighter", "Identify and highlight important terms and concepts"),
        ("🎙️ Voice Assistant", "Text-to-speech and speech-to-text capabilities"),
        ("🌐 Multilingual Support", "Translate content into different languages"),
        ("🧠 Mind Maps", "Visualize concepts and relationships as mind maps"),
        ("⏱️ Study Timer", "Pomodoro timer and progress tracking")
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
                
                # Set up FAISS index
                st.session_state.faiss_index, st.session_state.embeddings = setup_faiss_index(st.session_state.chunks)
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
            # Find relevant chunks
            relevant_chunks = find_relevant_chunks(
                question, 
                st.session_state.chunks, 
                st.session_state.faiss_index,
                st.session_state.embeddings
            )
            
            # Generate answer based on difficulty
            prompt = f"Provide a {difficulty.lower()} level explanation: {question}"
            answer = generate_content(prompt, relevant_chunks)
            
            # Display answer
            st.markdown("### Answer")
            st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)
            
            # Voice output option
            if st.button("Read Aloud") and TTS_AVAILABLE:
                audio_bytes = text_to_speech(answer)
                if audio_bytes:
                    st.audio(audio_bytes, format='audio/mp3')
                else:
                    st.warning("Text-to-speech is not working properly")
            
            # Show context sources
            with st.expander("View source passages"):
                for i, chunk in enumerate(relevant_chunks):
                    st.markdown(f"**Passage {i+1}:**")
                    st.write(chunk)
                    st.markdown("---")

# Smart Summarizer
elif app_mode == "Smart Summarizer" and st.session_state.processed:
    st.markdown("### Generate Smart Summaries")
    
    summary_length = st.select_slider("Summary Length", 
                                     options=["Very Short", "Short", "Medium", "Detailed"],
                                     value="Medium")
    
    if st.button("Generate Summary"):
        with st.spinner("Creating your summary..."):
            # Use all chunks for a comprehensive summary
            prompt = f"Create a {summary_length.lower()} summary of the key points"
            summary = generate_content(prompt, st.session_state.chunks, max_length=1000)
            
            st.markdown("### Document Summary")
            st.markdown(f'<div class="answer-box">{summary}</div>', unsafe_allow_html=True)
            
            # Download option
            st.download_button("Download Summary", summary, file_name="document_summary.txt")

# Flashcards Generator
elif app_mode == "Flashcards Generator" and st.session_state.processed:
    st.markdown("### Generate Study Flashcards")
    
    num_flashcards = st.slider("Number of Flashcards", 3, 20, 5)
    
    if st.button("Generate Flashcards"):
        with st.spinner("Creating your flashcards..."):
            # Generate flashcards
            flashcards = []
            for i in range(num_flashcards):
                prompt = f"Create flashcard {i+1} with a question and answer"
                flashcard_content = generate_content(prompt, st.session_state.chunks)
                
                # Parse the simulated flashcard content
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
        cols = st.columns(2)
        for i, card in enumerate(flashcards):
            with cols[i % 2]:
                st.markdown(f"""
                <div class="flashcard">
                    <div class="flashcard-inner">
                        <div class="flashcard-front">
                            <h3>{card['question']}</h3>
                        </div>
                        <div class="flashcard-back">
                            <p>{card['answer']}</p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Download option
        flashcard_text = "\n\n".join([f"Q: {card['question']}\nA: {card['answer']}" for card in flashcards])
        st.download_button("Download Flashcards", flashcard_text, file_name="flashcards.txt")

# Quiz Maker
elif app_mode == "Quiz Maker" and st.session_state.processed:
    st.markdown("### Generate Practice Quiz")
    
    quiz_type = st.radio("Quiz Type", ("Multiple Choice", "True/False"))
    num_questions = st.slider("Number of Questions", 3, 15, 5)
    
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
            st.markdown(f"**Question {i+1}:**")
            st.write(question)
            st.text_input(f"Your answer for question {i+1}", key=f"quiz_ans_{i}")
            st.markdown("---")
        
        if st.button("Check Answers"):
            st.info("In a full implementation, this would validate your answers against the correct ones.")

# Voice Assistant
elif app_mode == "Voice Assistant" and st.session_state.processed:
    st.markdown("### Voice Assistant")
    
    if not TTS_AVAILABLE:
        st.warning("Text-to-speech functionality is not available. Please install gTTS with: pip install gTTS")
    else:
        st.info("Select text to convert to speech")
        
        text_to_read = st.text_area("Enter text to read aloud", height=100,
                                  value="Welcome to StudyMate Pro. This is a text-to-speech demonstration.")
        
        if st.button("Read Text Aloud"):
            audio_bytes = text_to_speech(text_to_read)
            if audio_bytes:
                st.audio(audio_bytes, format='audio/mp3')
                st.success("Audio generated successfully!")
            else:
                st.error("Failed to generate audio")

# Multilingual Support
elif app_mode == "Multilingual Support" and st.session_state.processed:
    st.markdown("### Multilingual Support")
    
    if not TRANSLATION_AVAILABLE:
        st.warning("Translation functionality is not available. Please install googletrans with: pip install googletrans==4.0.0rc1")
    else:
        languages = {
            "English": "en",
            "Spanish": "es",
            "French": "fr",
            "German": "de",
            "Hindi": "hi",
            "Tamil": "ta"
        }
        
        selected_lang = st.selectbox("Select target language", list(languages.keys()))
        
        if st.button("Translate Content"):
            with st.spinner("Translating content..."):
                # Use a sample of the content for translation
                sample_text = st.session_state.chunks[0][:200] + "..." if st.session_state.chunks else "No content available"
                translated = translate_text(sample_text, languages[selected_lang])
                
                st.markdown("### Translated Content")
                st.markdown(f'<div class="answer-box">{translated}</div>', unsafe_allow_html=True)

# Mind Map Generator
elif app_mode == "Mind Map Generator" and st.session_state.processed:
    st.markdown("### Mind Map Generator")
    
    if st.button("Generate Mind Map"):
        with st.spinner("Creating mind map..."):
            mind_map = generate_mind_map(st.session_state.chunks)
            
            # Create visualization
            plt.figure(figsize=(10, 8))
            pos = nx.spring_layout(mind_map, k=0.5, iterations=50)
            nx.draw(mind_map, pos, with_labels=True, node_color='skyblue', 
                    node_size=2000, font_size=10, font_weight='bold')
            plt.title("Concept Mind Map")
            
            st.pyplot(plt)
            st.info("This is a simplified visualization. A full implementation would create a more detailed concept map.")

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
        <p>Python, Streamlit, FAISS, SentenceTransformers, and more</p>
    </div>
""", unsafe_allow_html=True)




