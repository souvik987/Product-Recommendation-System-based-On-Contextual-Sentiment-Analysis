import os
import re
import wave
import streamlit as st
import pandas as pd
import json
import time
import requests
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import speech_recognition as sr
from PIL import Image
import io
import base64
import tempfile
import streamlit.components.v1 as components
import queue
import threading
import numpy as np
from typing import Any, Dict, Tuple
from typing import List
import logging

from recommend import MultimodalSentimentAnalyzer

def prepare_multimodal_input(text_input, voice_input, image_input):
    """Prepare inputs for multimodal analysis"""
    inputs = {}
    modalities = []
    
    if text_input and text_input.strip():
        inputs['text'] = text_input.strip()
        modalities.append('text')
    
    if voice_input and voice_input.strip():
        inputs['voice'] = voice_input.strip()
        # If we have voice, it should be the primary text source
        if 'text' not in inputs:
            inputs['text'] = voice_input.strip()
        modalities.append('voice')
    
    if image_input:
        inputs['image'] = image_input
        modalities.append('image')
    
    inputs['modalities_used'] = modalities
    return inputs

# Import your existing ML system (assuming it's in the same directory)
# Make sure to adjust the import path based on your project structure
try:
    from recommend import EnhancedRecommendationSystem, WEATHER_API_KEY
    ML_SYSTEM_AVAILABLE = True
except ImportError:
    ML_SYSTEM_AVAILABLE = False
    st.error("ML Recommendation System not found. Please ensure 'paste.py' is in the same directory.")


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Streamlit page
st.set_page_config(
    page_title="Smart Product Recommender",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)   

# Custom CSS for better styling
st.markdown("""
<style>
    section[data-testid="stSidebar"] {
    overflow-y: auto !important;
    max-height: 100vh !important;
    }
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 2rem;
    }
    .weather-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .recommendation-card {
        background: #f8fafc;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #3b82f6;
        margin: 0.5rem 0;
    }
    .sentiment-positive {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        text-align: center;
    }
    .sentiment-negative {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        text-align: center;
    }
    .sentiment-neutral {
        background: linear-gradient(135deg, #6b7280 0%, #4b5563 100%);
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        text-align: center;
    }
    .location-status {
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
        text-align: center;
    }
    .location-success {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .location-error {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    .location-loading {
        background-color: #d1ecf1;
        color: #0c5460;
        border: 1px solid #bee5eb;
    }   
     .voice-recording {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .voice-transcription {
        background: #f0f9ff;
        border: 2px dashed #3b82f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .voice-recommendation-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 2rem 0;
        color: white;
    }
    .voice-input-ready {
        background: #dcfce7;
        border: 2px solid #16a34a;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .recommendation-container {
        margin-top: 2rem;
        padding: 1rem;
        border-top: 3px solid #3b82f6;
    }     
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'recommender' not in st.session_state and ML_SYSTEM_AVAILABLE:
    st.session_state.recommender = EnhancedRecommendationSystem(WEATHER_API_KEY)
    # Load the dataset - adjust path as needed
    try:
        st.session_state.recommender.load_data('flipkart_com-ecommerce_sample.csv')
    except FileNotFoundError:
        st.error("Product dataset not found. Please ensure 'flipkart_com-ecommerce_sample.csv' is available.")
    
    # Enable sentiment analysis features
    st.session_state.recommender.sentiment_analyzer.enable_image_analysis("mock_key")
    st.session_state.recommender.sentiment_analyzer.enable_voice_analysis("mock_key")

# Add this function to handle component communication
def handle_component_communication():
    """Handle communication from HTML components"""
    # Check if there's any component data in query params or other sources
    query_params = st.query_params
    
    # Check for voice transcription in query params
    if 'voice_transcription' in query_params:
        voice_text = query_params['voice_transcription'][0]
        if voice_text and len(voice_text.strip()) > 3:
            st.session_state.voice_transcription = voice_text.strip()
            # Clear the query param to prevent repeated processing
            st.query_params.clear()

def initialize_all_session_state():
    """Initialize all session state variables properly"""

    if 'user_id' not in st.session_state:
        st.session_state.user_id = f"user_{int(time.time())}"

    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = None

    if 'user_location' not in st.session_state:
        st.session_state.user_location = "Kolkata"

    if 'location_data' not in st.session_state:
        st.session_state.location_data = None

    if 'location_method' not in st.session_state:
        st.session_state.location_method = "IP-based"

    if 'voice_transcription' not in st.session_state:
        st.session_state.voice_transcription = ""

    if 'voice_transcription_history' not in st.session_state:
        st.session_state.voice_transcription_history = []

    if 'last_voice_transcription' not in st.session_state:
        st.session_state.last_voice_transcription = ""

    if 'voice_ready_for_recommendations' not in st.session_state:
        st.session_state.voice_ready_for_recommendations = False

    if 'pending_voice_transcription' not in st.session_state:
        st.session_state.pending_voice_transcription = ""

    if 'manual_voice_input' not in st.session_state:
        st.session_state.manual_voice_input = ""
    
    if 'voice_transcription' not in st.session_state:
        st.session_state.voice_transcription = ""
    if 'voice_recommendations_generated' not in st.session_state:
        st.session_state.voice_recommendations_generated = False
    if 'voice_processor' not in st.session_state:
        st.session_state.voice_processor = None
    
    # Add voice component communication handling
    if 'voice_component_data' not in st.session_state:
        st.session_state.voice_component_data = None

    # Handle component communication
    handle_component_communication()   

initialize_all_session_state()    

# Voice Recording Classes
class SimplifiedVoiceProcessor:
    """Simplified voice processor with better state management"""
    
    def __init__(self):
        self.sample_rate = 16000
        self.min_audio_length = 1.0
        
    def preprocess_voice_text(self, voice_text: str) -> str:
        """Clean and preprocess voice text for recommendations"""
        if not voice_text:
            return ""
        
        # Clean the voice text
        cleaned_text = voice_text.strip().lower()
        
        # Remove filler words
        filler_words = ['um', 'uh', 'like', 'you know', 'actually', 'basically', 'so', 'well', 'okay', 'alright']
        for filler in filler_words:
            cleaned_text = re.sub(r'\b' + filler + r'\b', '', cleaned_text)
        
        # Remove extra whitespace
        cleaned_text = ' '.join(cleaned_text.split())
        
        # Common voice-to-text corrections
        corrections = {
            'show me': 'find',
            'i want': 'find',
            'i need': 'find',
            'looking for': 'find',
            'search for': 'find',
            'mobile': 'phone',
            'cell phone': 'phone',
            'laptop computer': 'laptop',
            'running shoes': 'shoes sport',
            'womens': 'women',
            'mens': 'men'
        }
        
        for old, new in corrections.items():
            cleaned_text = cleaned_text.replace(old, new)
        
        return cleaned_text if cleaned_text else voice_text

def create_voice_input_component():
    """Create a more reliable voice input component"""
    
    component_html = """
    <div style="text-align: center; padding: 20px; border: 2px solid #3b82f6; border-radius: 10px; margin: 10px 0; background: linear-gradient(135deg, #f8fafc, #e2e8f0);">
        <h4 style="color: #1e40af; margin-bottom: 15px;">🎙️ Voice Input</h4>
        
        <div style="margin: 15px 0;">
            <button id="startBtn" onclick="startRecording()" style="
                background: linear-gradient(135deg, #10b981, #059669);
                color: white;
                padding: 12px 25px;
                border: none;
                border-radius: 20px;
                cursor: pointer;
                font-size: 16px;
                margin: 5px;
                font-weight: bold;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                transition: all 0.3s ease;
            ">
                🎤 Start Recording
            </button>
            
            <button id="stopBtn" onclick="stopRecording()" disabled style="
                background: #6b7280;
                color: white;
                padding: 12px 25px;
                border: none;
                border-radius: 20px;
                cursor: not-allowed;
                font-size: 16px;
                margin: 5px;
                font-weight: bold;
            ">
                ⏹️ Stop Recording
            </button>
        </div>
        
        <div id="status" style="
            margin: 10px 0; 
            font-weight: bold; 
            color: #059669;
            padding: 8px;
            border-radius: 5px;
            background: #ecfdf5;
        ">Ready to record</div>
        
        <div id="transcription" style="
            margin: 10px 0; 
            padding: 12px; 
            background: #f0f9ff; 
            border: 1px solid #3b82f6;
            border-radius: 6px; 
            min-height: 60px; 
            text-align: left;
            font-family: -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 14px;
        ">Click "Start Recording" to begin...</div>
        
        <!-- Hidden input to store transcription for Streamlit -->
        <input type="hidden" id="voice_result" value="">
    </div>

    <script>
        let recognition;
        let isRecording = false;
        let finalTranscript = '';
        
        function updateStatus(message, color = '#059669', bgColor = '#ecfdf5') {
            const statusDiv = document.getElementById('status');
            statusDiv.style.color = color;
            statusDiv.style.backgroundColor = bgColor;
            statusDiv.innerHTML = message;
        }

        function sendTranscriptionToStreamlit(text) {
            if (text && text.trim().length > 0) {
                const cleanText = text.trim();
                
                // Store in hidden input
                document.getElementById('voice_result').value = cleanText;
                
                // Store in sessionStorage
                sessionStorage.setItem('voice_transcription', cleanText);
                sessionStorage.setItem('voice_timestamp', Date.now().toString());
                
                console.log('Sending transcription to Streamlit:', cleanText);
                
                // CRITICAL FIX: Use Streamlit's component value system
                const data = {
                    transcription: cleanText,
                    timestamp: Date.now(),
                    success: true,
                    type: 'voice_transcription'
                };
                
                // Method 1: Direct Streamlit component communication
                if (window.Streamlit) {
                    window.Streamlit.setComponentValue(data);
                }
                
                // Method 2: PostMessage for broader compatibility
                window.parent.postMessage({
                    type: 'streamlit:setComponentValue',
                    value: data
                }, '*');
                
                // Method 3: Custom event for fallback
                window.parent.postMessage({
                    type: 'voice_transcription_complete',
                    transcription: cleanText,
                    timestamp: Date.now()
                }, '*');

                // Method 4: Force session storage update with ready flag
                sessionStorage.setItem('voice_transcription', cleanText);
                sessionStorage.setItem('voice_ready', 'true');
                sessionStorage.setItem('voice_timestamp', Date.now().toString());

                // Force a page interaction to trigger Streamlit update
                setTimeout(() => {
                    sessionStorage.setItem('voice_transcription', cleanText);
                    sessionStorage.setItem('voice_ready', 'true');
                    // Trigger a click event to force Streamlit to check
                    const event = new CustomEvent('voiceTranscriptionReady', {detail: cleanText});
                    window.dispatchEvent(event);
                }, 500);
                
                console.log('Transcription sent via all methods');
            }
        }

        function initSpeechRecognition() {
            if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
                updateStatus('❌ Speech recognition not supported in this browser', '#dc2626', '#fef2f2');
                document.getElementById('startBtn').disabled = true;
                return false;
            }

            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            recognition = new SpeechRecognition();
            
            recognition.continuous = true;
            recognition.interimResults = true;
            recognition.lang = 'en-US';
            
            recognition.onstart = function() {
                isRecording = true;
                finalTranscript = '';
                updateStatus('🔴 Recording... Speak now!', '#dc2626', '#fecaca');
                
                document.getElementById('startBtn').disabled = true;
                document.getElementById('stopBtn').disabled = false;
                document.getElementById('stopBtn').style.background = '#ef4444';
                document.getElementById('stopBtn').style.cursor = 'pointer';
                document.getElementById('transcription').innerHTML = '🎤 Listening...';
            };
            
            recognition.onresult = function(event) {
                let interimTranscript = '';
                finalTranscript = '';
                
                for (let i = 0; i < event.results.length; i++) {
                    if (event.results[i].isFinal) {
                        finalTranscript += event.results[i][0].transcript + ' ';
                    } else {
                        interimTranscript += event.results[i][0].transcript;
                    }
                }
                
                const display = finalTranscript + 
                    (interimTranscript ? '<span style="color: #6b7280; font-style: italic;">' + interimTranscript + '</span>' : '');
                
                document.getElementById('transcription').innerHTML = display || '🎤 Listening...';
            };
            
            recognition.onerror = function(event) {
                updateStatus('❌ Error: ' + event.error, '#dc2626', '#fef2f2');
                resetButtons();
            };
            
            recognition.onend = function() {
                isRecording = false;
                
                if (finalTranscript.trim()) {
                    const cleanTranscription = finalTranscript.trim();
                    updateStatus('✅ Voice captured! Processing...', '#059669', '#dcfce7');
                    document.getElementById('transcription').innerHTML = cleanTranscription;
                    
                    // CRITICAL: Send to Streamlit immediately
                    sendTranscriptionToStreamlit(cleanTranscription);
                    
                    setTimeout(() => {
                        updateStatus('✅ Voice ready for recommendations!', '#059669', '#dcfce7');
                    }, 1000);
                } else {
                    updateStatus('⚠️ No speech detected. Try again.', '#ca8a04', '#fefce8');
                    document.getElementById('transcription').innerHTML = 'No speech detected. Please try again.';
                }
                
                resetButtons();
            };
            
            return true;
        }

        function startRecording() {
            if (!recognition && !initSpeechRecognition()) {
                return;
            }
            
            try {
                recognition.start();
            } catch (error) {
                updateStatus('❌ Failed to start recording', '#dc2626', '#fef2f2');
                resetButtons();
            }
        }

        function stopRecording() {
            if (recognition && isRecording) {
                recognition.stop();
                updateStatus('⏳ Processing...', '#ca8a04', '#fefce8');
            }
        }

        function resetButtons() {
            document.getElementById('startBtn').disabled = false;
            document.getElementById('stopBtn').disabled = true;
            document.getElementById('stopBtn').style.background = '#6b7280';
            document.getElementById('stopBtn').style.cursor = 'not-allowed';
        }

        // Initialize on load
        document.addEventListener('DOMContentLoaded', function() {
            initSpeechRecognition();
            
            // Check for existing transcription
            const stored = sessionStorage.getItem('voice_transcription');
            if (stored && stored.trim().length > 3) {
                document.getElementById('transcription').innerHTML = stored;
                updateStatus('✅ Voice ready for recommendations!', '#059669', '#dcfce7');
            }
        });
    </script>
    """
    
    # CRITICAL FIX: Return the component data properly
    return st.components.v1.html(component_html, height=400)

def initialize_voice_session_state():
    """Initialize voice-related session state variables"""
    if 'voice_transcription' not in st.session_state:
        st.session_state.voice_transcription = ""
    if 'voice_recommendations_generated' not in st.session_state:
        st.session_state.voice_recommendations_generated = False
    if 'voice_processor' not in st.session_state or st.session_state.voice_processor is None:
        st.session_state.voice_processor = SimplifiedVoiceProcessor()

def process_voice_recommendations(voice_text: str):
    """Process voice input and generate recommendations"""
    try:
        if not voice_text or len(voice_text.strip()) < 3:
            st.error("❌ Voice input too short. Please speak more clearly.")
            return
        
        # Check if recommender system exists
        if 'recommender' not in st.session_state:
            st.error("❌ Recommendation system not initialized. Please restart the app.")
            return
        
        with st.spinner("🎤 Processing your voice request..."):
            # Get the processor and recommender
            processor = st.session_state.voice_processor
            recommender = st.session_state.recommender
            
            # Preprocess the voice text
            processed_query = processor.preprocess_voice_text(voice_text)
            
            # Show what we're processing
            st.info(f"🔍 Processing: '{processed_query}'")
            
            # Get user data from session state
            user_id = st.session_state.get('user_id', 'user_001')
            city = st.session_state.get('user_location', 'Kolkata')
            
            # Generate recommendations using your existing system
            results = recommender.get_enhanced_recommendations(
                user_id=user_id,
                city=city,
                user_text=processed_query,  # Use processed voice text
                user_image=None,
                user_voice=None
            )
            
            if results:
                # Store results in session state
                st.session_state.recommendations = results
                st.session_state.voice_recommendations_generated = True
                
                # Count recommendations
                rec_count = 0
                if isinstance(results, dict) and 'recommendations' in results:
                    recommendations = results['recommendations']
                    if hasattr(recommendations, '__len__'):
                        rec_count = len(recommendations)
                elif isinstance(results, list):
                    rec_count = len(results)
                
                st.success(f"✅ Generated {rec_count} voice-based recommendations!")
                
                # Add voice metadata to results
                if isinstance(results, dict):
                    results['voice_input'] = voice_text
                    results['processed_voice_query'] = processed_query
                    results['is_voice_generated'] = True
                
                # Clear the voice input to prevent duplicate processing
                st.session_state.voice_transcription = ""

                results['has_voice_input'] = True
                results['voice_text_used'] = voice_text
                
                # Force a rerun to show recommendations
                st.rerun()
                
            else:
                st.warning("⚠️ No recommendations found. Try speaking more specifically about what you're looking for.")
                
    except Exception as e:
        st.error(f"❌ Error processing voice recommendations: {str(e)}")
        logger.error(f"Voice recommendation error: {e}")

def render_voice_section():
    """Render the complete voice section with proper button handling"""
    
    # Initialize session state
    initialize_voice_session_state()
    
    st.markdown("## 🎙️ Voice-Based Recommendations")
    
    # Create the voice input component and capture its return value
    voice_data = create_voice_input_component()

    # CRITICAL FIX: Check browser storage immediately after component creation
    check_voice_storage = st.components.v1.html("""
    <script>
    const stored = sessionStorage.getItem('voice_transcription');
    if (stored && stored.trim().length > 3) {
        document.write(stored);
    }
    </script>
    """, height=0)

    if check_voice_storage and len(str(check_voice_storage).strip()) > 3:
        st.session_state.voice_transcription = str(check_voice_storage).strip()
        
    # CRITICAL FIX: Process voice data immediately if received
    if voice_data:
        try:
            # Handle different data formats
            if isinstance(voice_data, dict):
                if voice_data.get('success') and voice_data.get('transcription'):
                    transcription = voice_data['transcription'].strip()
                    if transcription and len(transcription) > 3:
                        st.session_state.voice_transcription = transcription
                        st.success(f"✅ Voice captured: {transcription}")
                       
            elif isinstance(voice_data, str) and len(voice_data.strip()) > 3:
                st.session_state.voice_transcription = voice_data.strip()
                st.rerun()
        except Exception as e:
            st.error(f"Error processing voice data: {str(e)}")
    
    # CRITICAL FIX: Always check session state for voice input
    current_voice_input = st.session_state.get('voice_transcription', '').strip()

    # Force check browser storage on every render
    if not current_voice_input:
        storage_check = st.components.v1.html("""
        <script>
        const stored = sessionStorage.getItem('voice_transcription');
        if (stored && stored.trim().length > 3) {
            document.write(JSON.stringify({transcription: stored, ready: true}));
        } else {
            document.write('{}');
        }
        </script>
        """, height=0)
        
        if storage_check:
            try:
                import json
                result = json.loads(str(storage_check))
                if result.get('transcription'):
                    current_voice_input = result['transcription'].strip()
                    st.session_state.voice_transcription = current_voice_input
                    st.rerun()
            except:
                pass
    
    # CRITICAL FIX: Show the recommendation section if there's ANY voice input
    # CRITICAL FIX: Show the recommendation section if there's ANY voice input
    if current_voice_input and len(current_voice_input.strip()) > 3:
        # Force update the session state
        st.session_state.voice_transcription = current_voice_input.strip()
        # Show success message
        st.success("✅ Voice input captured successfully!")
        
        # CRITICAL FIX: Make the button section more prominent
        st.markdown("### 🎯 Generate Recommendations")
        
        # Always show the button if we have voice input
        recommend_key = f"voice_recommend_{hash(current_voice_input)}"

        col1, col2 = st.columns([4, 1])
        with col1:
            recommend_button = st.button(
                "🎯 GET VOICE RECOMMENDATIONS NOW", 
                key=recommend_key,
                type="primary",
                use_container_width=True,
                help="Generate recommendations based on your voice input"
            )
            
            # Process recommendations when button is clicked
            if recommend_button:
                # Ensure we have the latest voice input
                final_voice_input = current_voice_input or st.session_state.voice_transcription
                if final_voice_input and len(final_voice_input.strip()) > 3:
                    process_voice_recommendations(final_voice_input)
                else:
                    st.error("No valid voice input found. Please record again.")

        with col2:
            if st.button("🗑️ Clear", key=f"voice_clear_{int(time.time())}", use_container_width=True):
                # Clear session state
                st.session_state.voice_transcription = ""
                st.session_state.voice_recommendations_generated = False

                # 🔧 Add this: mark cleared state to avoid rereading
                st.session_state.voice_component_data = None

                # Clear browser sessionStorage with JS + force page reload
                clear_storage_html = """
                <script>
                    sessionStorage.removeItem('voice_transcription');
                    sessionStorage.removeItem('voice_timestamp');
                    sessionStorage.removeItem('voice_ready');
                    sessionStorage.removeItem('voice_data_ready');
                    location.reload();  // force page reload to sync frontend state
                </script>
                """
                st.components.v1.html(clear_storage_html, height=0)
    
    else:
        # Show instructions when no voice input
        st.markdown("""
        <div style="
            background: #f8fafc; 
            border: 2px dashed #6b7280; 
            padding: 20px; 
            border-radius: 10px; 
            text-align: center;
            margin: 15px 0;
        ">
            <h4 style="color: #374151; margin-bottom: 10px;">🎤 Ready for Voice Input</h4>
            <p style="color: #6b7280; margin: 0;">
                Click "Start Recording" above and speak your requirements.<br>
                The recommendation button will appear after processing your voice.
            </p>
        </div>
        """, unsafe_allow_html=True)
        

# MAIN INTEGRATION FUNCTION - Replace your existing voice code with this:
def integrate_voice_system():
    """Main function to integrate the voice system into your sidebar"""
    
    with st.sidebar:
        st.markdown("---")
        render_voice_section()
        st.markdown("---")


# Function to get location using IP-based geolocation
def get_location_from_ip():
    """Get user location using IP-based geolocation services"""
    try:
        # Method 1: Using ipapi.co (free, no API key required)
        response = requests.get('https://ipapi.co/json/', timeout=5)
        if response.status_code == 200:
            data = response.json()
            return {
                'city': data.get('city', 'Unknown'),
                'country': data.get('country_name', 'Unknown'),
                'latitude': data.get('latitude'),
                'longitude': data.get('longitude'),
                'method': 'IP Geolocation',
                'success': True
            }
    except Exception as e:
        pass
    
    try:
        # Method 2: Using ipinfo.io (backup)
        response = requests.get('https://ipinfo.io/json', timeout=5)
        if response.status_code == 200:
            data = response.json()
            location = data.get('loc', '').split(',')
            return {
                'city': data.get('city', 'Unknown'),
                'country': data.get('country', 'Unknown'),
                'latitude': float(location[0]) if len(location) > 0 else None,
                'longitude': float(location[1]) if len(location) > 1 else None,
                'method': 'IP Geolocation',
                'success': True
            }
    except Exception as e:
        pass
    
    return {
        'city': 'Kolkata',
        'country': 'India',
        'latitude': 22.5726,
        'longitude': 88.3639,
        'method': 'Default (Fallback)',
        'success': False
    }

# JavaScript code for browser-based geolocation
def get_browser_location_component():
    """Create a component for browser-based geolocation"""
    return components.html("""
    <script>
        function getLocation() {
            if (navigator.geolocation) {
                navigator.geolocation.getCurrentPosition(
                    function(position) {
                        const data = {
                            latitude: position.coords.latitude,
                            longitude: position.coords.longitude,
                            accuracy: position.coords.accuracy,
                            timestamp: new Date().toISOString(),
                            method: 'Browser GPS'
                        };
                        
                        // Send data back to Streamlit
                        window.parent.postMessage({
                            type: 'location_data',
                            data: data
                        }, '*');
                        
                        document.getElementById('status').innerHTML = 
                            '<div class="location-success">✅ Location detected successfully!</div>';
                    },
                    function(error) {
                        let errorMsg = 'Location access denied or failed';
                        switch(error.code) {
                            case error.PERMISSION_DENIED:
                                errorMsg = "Location access denied by user";
                                break;
                            case error.POSITION_UNAVAILABLE:
                                errorMsg = "Location information unavailable";
                                break;
                            case error.TIMEOUT:
                                errorMsg = "Location request timed out";
                                break;
                        }
                        
                        document.getElementById('status').innerHTML = 
                            '<div class="location-error">❌ ' + errorMsg + '</div>';
                        
                        window.parent.postMessage({
                            type: 'location_error',
                            error: errorMsg
                        }, '*');
                    },
                    {
                        enableHighAccuracy: true,
                        timeout: 10000,
                        maximumAge: 60000
                    }
                );
            } else {
                document.getElementById('status').innerHTML = 
                    '<div class="location-error">❌ Geolocation not supported by browser</div>';
            }
        }
        
        // Auto-start location detection
        document.addEventListener('DOMContentLoaded', function() {
            document.getElementById('status').innerHTML = 
                '<div class="location-loading">🔍 Detecting location...</div>';
            getLocation();
        });
    </script>
    
    <div id="status"></div>
    <button onclick="getLocation()" style="
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        font-size: 14px;
    ">🔄 Retry Location Detection</button>
    """, height=100)

# Function to convert coordinates to city name
def get_city_from_coordinates(lat, lon):
    """Convert latitude and longitude to city name using reverse geocoding"""
    try:
        # Using OpenStreetMap Nominatim (free, no API key required)
        url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}&zoom=10&addressdetails=1"
        headers = {'User-Agent': 'StreamlitApp/1.0'}
        
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            data = response.json()
            address = data.get('address', {})
            
            city = (address.get('city') or 
                   address.get('town') or 
                   address.get('village') or 
                   address.get('municipality') or 
                   'Unknown')
            
            return {
                'city': city,
                'country': address.get('country', 'Unknown'),
                'state': address.get('state', 'Unknown'),
                'success': True
            }
    except Exception as e:
        st.error(f"Reverse geocoding error: {str(e)}")
    
    return {'city': 'Unknown', 'country': 'Unknown', 'state': 'Unknown', 'success': False}


# Main header
st.markdown('<h1 class="main-header">🛍️ Smart Product Recommender</h1>', unsafe_allow_html=True)

# Sidebar for user inputs
with st.sidebar:
    st.header("🎯 Personalization Settings")
    
    # Location input
    st.subheader("📍 Location")
    location_method = st.radio(
        "Choose location detection method:",
        ["Automatic (IP-based)", "Browser GPS", "Manual Entry"],
        help="IP-based is fastest, Browser GPS is most accurate, Manual Entry gives you full control"
    )
    
    if location_method == "Automatic (IP-based)":
        if st.button("🔍 Detect Location Automatically"):
            with st.spinner("Detecting your location..."):
                location_data = get_location_from_ip()
                st.session_state.location_data = location_data
                
                if location_data['success']:
                    st.session_state.user_location = location_data['city']
                    st.success(f"📍 Location detected: {location_data['city']}, {location_data['country']}")
                    st.info(f"Method: {location_data['method']}")
                else:
                    st.warning("Using default location: Kolkata")
                    st.session_state.user_location = "Kolkata"
        
        # Show current location
        if st.session_state.location_data:
            loc_data = st.session_state.location_data
            st.markdown(f"""
            **Current Location:**
            - 🏙️ City: {loc_data['city']}
            - 🌍 Country: {loc_data['country']}
            - 📡 Method: {loc_data['method']}
            """)
    
    elif location_method == "Browser GPS":
        st.info("🌐 Browser GPS provides the most accurate location but requires permission")
        
        # Browser-based geolocation component
        get_browser_location_component()
        
        # Manual coordinate input for browser GPS results
        st.subheader("GPS Coordinates (if detected)")
        col1, col2 = st.columns(2)
        with col1:
            latitude = st.number_input("Latitude", value=22.5726, format="%.6f")
        with col2:
            longitude = st.number_input("Longitude", value=88.3639, format="%.6f")
        
        if st.button("🔄 Convert Coordinates to City"):
            with st.spinner("Converting coordinates to city name..."):
                city_data = get_city_from_coordinates(latitude, longitude)
                if city_data['success']:
                    st.session_state.user_location = city_data['city']
                    st.success(f"📍 Location: {city_data['city']}, {city_data['state']}, {city_data['country']}")
                else:
                    st.error("Could not determine city from coordinates")
    
    else:  # Manual Entry
        user_location = st.text_input("Enter your city:", value=st.session_state.user_location)
        st.session_state.user_location = user_location
        
        # Optional: Validate city name
        if st.button("✅ Validate City"):
            if user_location.strip():
                st.success(f"✅ City set to: {user_location}")
            else:
                st.error("Please enter a valid city name")
    
    # Display current location
    st.markdown(f"**Current Location:** {st.session_state.user_location}")
    
    st.divider()
    
    # Text input for sentiment analysis
    st.subheader("💭 Tell us about your mood/needs")
    user_text = st.text_area(
        "Describe what you're looking for or how you're feeling:",
        placeholder="I'm looking for something comfortable for rainy weather...",
        height=100
    )
    
    st.divider()
    
    # Image upload
    st.subheader("📷 Upload an image (optional)")
    uploaded_image = st.file_uploader(
        "Upload an image that represents your style or mood:",
        type=['png', 'jpg', 'jpeg'],
        help="This helps us understand your preferences better"
    )

    
    st.divider()

    # VOICE SYSTEM INTEGRATION - THIS IS THE NEW PART
    integrate_voice_system()
        
    st.divider()
    
    
    # Activity simulation
    st.subheader("🔄 Simulate User Activity")
    if st.button("Simulate Browsing Activity"):
        if ML_SYSTEM_AVAILABLE:
            # Simulate some user activities
            recommender = st.session_state.recommender
            user_id = st.session_state.user_id
            
            # Simulate browsing
            recommender.user_tracker.track_activity(user_id, 'browsing', {
                'url': '/summer-collection',
                'time_spent': 180,
                'scroll_depth': 0.75,
                'clicks': 12,
                'mouse_movement': 1500
            })
            
            # Simulate product view
            recommender.user_tracker.track_activity(user_id, 'product_view', {
                'product_id': 'P123',
                'category': 'Summer wear',
                'time_spent': 45
            })
            
            # Simulate search
            recommender.user_tracker.track_activity(user_id, 'search', {
                'query': 'waterproof boots'
            })
            
            st.success("User activity simulated!")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("🎯 Get Personalized Recommendations")
    
    # Generate recommendations button
    if st.button("🚀 Generate Recommendations", type="primary", use_container_width=True):
        if 'recommender' not in st.session_state:
                st.error("Please initialize the recommendation system first")
        else:
            with st.spinner("Generating recommendations..."):
                # Check if we have voice input
                current_voice_input = st.session_state.get('voice_transcription', '').strip()
                
                # Prepare voice parameter - this is the key fix
                voice_input_for_analysis = None
                if st.session_state.get("voice_recommendations_generated", False):
                    voice_input_for_analysis = st.session_state.get("voice_transcription", "").strip()
                
                # Save uploaded image temporarily if provided
                image_path = None
                if uploaded_image:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                        tmp_file.write(uploaded_image.read())
                        image_path = tmp_file.name
                else:
                    image_path = None
                
                try:
                    results = st.session_state.recommender.get_enhanced_recommendations(
                        user_id=st.session_state.user_id,
                        city=st.session_state.user_location,
                        user_text=user_text if user_text.strip() else None,
                        user_image=image_path,
                        user_voice=voice_input_for_analysis  # Pass voice input here
                    )
                    
                    if results:
                        st.session_state.recommendations = results
                        
                        # Add metadata to track if voice was used
                        if isinstance(results, dict) and voice_input_for_analysis:
                            results['has_voice_input'] = True
                            results['voice_text_used'] = voice_input_for_analysis
                        
                        st.success("✅ Recommendations generated!")
                        st.rerun()
                    
                    # Clean up temporary image file
                    if image_path and os.path.exists(image_path):
                        os.unlink(image_path)
                        
                except Exception as e:
                    st.error(f"Error generating recommendations: {str(e)}")
    
    # Display recommendations
    if st.session_state.recommendations:
        results = st.session_state.recommendations
        
        # Check if this is voice-based recommendations
        if hasattr(st.session_state, 'voice_recommendations_data') and st.session_state.voice_recommendations_data:
            voice_data = st.session_state.voice_recommendations_data
            st.info(f"🎤 Recommendations based on voice input: '{voice_data.get('original_voice_text', st.session_state.voice_transcription)}'")
        
        
        # Weather information - with error handling
        st.subheader("🌤️ Current Weather Context")
    
        try:
            # Safe weather extraction
            weather = None
            if isinstance(results, dict) and 'weather' in results:
                weather = results['weather']
            elif isinstance(results, list) and len(results) > 0 and isinstance(results[0], dict):
                weather = results[0].get('weather')
            
            if weather and isinstance(weather, dict):
                weather_col1, weather_col2, weather_col3, weather_col4 = st.columns(4)
                with weather_col1:
                    st.metric("Temperature", f"{weather.get('temperature', 'N/A')}°C")
                with weather_col2:
                    st.metric("Humidity", f"{weather.get('humidity', 'N/A')}%")
                with weather_col3:
                    st.metric("Wind Speed", f"{weather.get('wind_speed', 'N/A')} m/s")
                with weather_col4:
                    st.metric("Condition", weather.get('condition', 'N/A').title())
                
                st.markdown(f"**Description:** {weather.get('description', 'No description available').title()}")
            else:
                st.info("💡 Weather data not available - showing general recommendations")
                
        except Exception as e:
            st.warning("Weather information not available for voice-based recommendations")
            st.info("💡 Continuing with recommendations without weather context")
        
        # Context factors
        if 'context_factors' in results:
            st.subheader("🌍 Contextual Insights")
            context = results['context_factors']
            
            context_col1, context_col2 = st.columns(2)
            with context_col1:
                st.info(f"**Season:** {context['season']['season']}")
                st.write("**Seasonal Trends:**")
                for trend in context['season']['trends']:
                    st.write(f"• {trend}")
            
            with context_col2:
                st.info(f"**Region:** {context['region']}")
                st.write("**Regional Preferences:**")
                for pref in context['regional_preferences'][:5]:  # Show top 5
                    st.write(f"• {pref}")
        
        # Sentiment analysis results
        if 'sentiment_analysis' in results and results['sentiment_analysis']:
            st.subheader("😊 Sentiment Analysis")
            sentiment = results['sentiment_analysis']
            
            sentiment_col1, sentiment_col2 = st.columns(2)
            with sentiment_col1:
                # Fix: Show proper modalities based on what was actually analyzed
                modalities_used = sentiment.get('modalities_used', [])
                
                # Check if voice was actually used in this analysis
                has_voice = results.get('has_voice_input', False)
                voice_text_used = results.get('voice_text_used', '')
                
                # Display modalities more accurately
                if has_voice and voice_text_used:
                    if 'text' in modalities_used and len(modalities_used) == 1:
                        # Voice was converted to text, so show both
                        display_modalities = ['voice (converted to text)']
                    else:
                        display_modalities = modalities_used
                else:
                    display_modalities = modalities_used
                
                if not has_voice and 'voice' in modalities_used:
                    modalities_used = [m for m in modalities_used if m != 'voice']

                if modalities_used:
                    st.write(f"**Modalities Analyzed:** {', '.join(modalities_used)}")
                
                
                if 'combined_sentiment' in sentiment:
                    sentiment_score = sentiment['combined_sentiment']
                    if sentiment_score > 0.1:
                        st.markdown('<div class="sentiment-positive">😊 Positive Mood Detected</div>', 
                                unsafe_allow_html=True)
                    elif sentiment_score < -0.1:
                        st.markdown('<div class="sentiment-negative">😔 Negative Mood Detected</div>', 
                                unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="sentiment-neutral">😐 Neutral Mood</div>', 
                                unsafe_allow_html=True)
            
            with sentiment_col2:
                if 'text_sentiment' in sentiment:
                    text_sent = sentiment['text_sentiment']
                    sentiment_fig = go.Figure(data=go.Scatterpolar(
                        r=[text_sent['pos'], text_sent['neu'], text_sent['neg'], text_sent['compound']],
                        theta=['Positive', 'Neutral', 'Negative', 'Overall'],
                        fill='toself',
                        name='Voice/Text Sentiment' if has_voice else 'Text Sentiment'
                    ))
                    sentiment_fig.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[-1, 1])),
                        showlegend=False,
                        height=300
                    )
                    st.plotly_chart(sentiment_fig, use_container_width=True)


# Check if these are voice-based recommendations
        if hasattr(st.session_state, 'voice_recommendations_data'):
            voice_data = st.session_state.voice_recommendations_data
            st.info(f"🎤 Recommendations based on voice input: '{voice_data['original_voice_text']}'")
        
        # Show voice analysis details
            with st.expander("🔍 Voice Analysis Details"):
                col1, col2 = st.columns(2)
            
                with col1:
                    st.write("**Original Voice Input:**")
                    st.write(voice_data['original_voice_text'])
                    st.write("**Processed Query:**")
                    st.write(voice_data['processed_query'])
            
                with col2:
                    if 'sentiment_analysis' in voice_data:
                        sentiment = voice_data['sentiment_analysis']
                        st.write("**Sentiment Analysis:**")
                        st.write(f"Compound: {sentiment['compound']:.2f}")
                        st.write(f"Positive: {sentiment['pos']:.2f}")
                        st.write(f"Negative: {sentiment['neg']:.2f}")
                        st.write(f"Neutral: {sentiment['neu']:.2f}")
    
        
        # Top recommendations
        st.subheader("🎁 Your Personalized Recommendations")
        
        if isinstance(results, list):
            # If results is a list, assume first item contains recommendations
            if len(results) > 0 and isinstance(results[0], dict) and 'recommendations' in results[0]:
                recommendations_df = results[0]['recommendations']
            else:
                recommendations_df = pd.DataFrame()  # Empty DataFrame as fallback
        elif isinstance(results, dict) and 'recommendations' in results:
            recommendations_df = results['recommendations']
        else:
            recommendations_df = pd.DataFrame()
        
        if not recommendations_df.empty:
            # Display top 10 recommendations
            top_recommendations = recommendations_df.head(10)
            
            for idx, (_, product) in enumerate(top_recommendations.iterrows(), 1):
                with st.container():
                    rec_col1, rec_col2, rec_col3 = st.columns([1, 3, 1])
                    
                    with rec_col1:
                        st.markdown(f"### #{idx}")
                        if 'final_score' in product:
                            st.metric("Score", f"{product['final_score']:.2f}")
                    
                    with rec_col2:
                        st.markdown(f"**{product['product_name']}**")
                        st.write(f"*Category:* {product['product_category_tree']}")

                        rating_info = []
                        if 'product_rating' in product and pd.notna(product['product_rating']):
                            rating_info.append(f"⭐ Product Rating: {product['product_rating']}")
                        if 'overall_rating' in product and pd.notna(product['overall_rating']):
                            rating_info.append(f"🌟 Overall Rating: {product['overall_rating']}")
                        
                        if rating_info:
                            st.write(" | ".join(rating_info))
                    
                    with rec_col3:
                        if st.button(f"View Details", key=f"view_{idx}"):
                            st.session_state[f"show_details_{idx}"] = True
                    
                    # Show details if button clicked
                    if st.session_state.get(f"show_details_{idx}", False):
                        with st.expander("Product Details", expanded=True):
                            st.json(product.to_dict())
                    
                    st.divider()
        else:
            st.warning("No recommendations found. Try adjusting your inputs or location.")


    # Alternative fix for nested expander - use a simple text display:
    if st.session_state.voice_transcription_history:
        st.write("**Recent Voice Searches:**")
        for entry in reversed(st.session_state.voice_transcription_history[-3:]):  # Show last 3
            st.text(f"{entry['timestamp']}: {entry['text']}")

with col2:
    st.header("📊 Analytics Dashboard")

    # Location information
    st.subheader("📍 Location Info")
    if st.session_state.location_data:
        loc_data = st.session_state.location_data
        st.metric("Current City", loc_data['city'])
        st.metric("Country", loc_data['country'])
        st.info(f"Detection Method: {loc_data['method']}")
    else:
        st.info(f"Current Location: {st.session_state.user_location}")
    
    # User activity summary
    if ML_SYSTEM_AVAILABLE and st.session_state.recommendations:
        if isinstance(st.session_state.recommendations, dict):
            activity_data = st.session_state.recommendations.get('activity')
        else:
            activity_data = None
        
        if activity_data:
            st.subheader("👤 Your Activity")
            
            # Session metrics
            st.metric("Session Duration", f"{activity_data['session_duration']:.1f} min")
            st.metric("Activities", activity_data['activity_count'])
            st.metric("Cart Items", activity_data['cart_items'])
            
            # Category interests
            if activity_data['category_interests']:
                st.subheader("📈 Your Interests")
                interests_df = pd.DataFrame(
                    list(activity_data['category_interests'].items()),
                    columns=['Category', 'Interest_Level']
                )
                
                fig = px.bar(
                    interests_df, 
                    x='Interest_Level', 
                    y='Category',
                    orientation='h',
                    title="Category Interest Levels"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Recent searches
            if activity_data['search_history']:
                st.subheader("🔍 Recent Searches")
                for search in activity_data['search_history']:
                    st.write(f"• {search}")
    
    # System status
    st.subheader("⚡ System Status")
    if ML_SYSTEM_AVAILABLE:
        st.success("✅ ML System Active")
        st.success("✅ Weather API Connected")
        st.success("✅ Sentiment Analysis Ready")
        st.success("✅ Location Detection Ready")
        st.success("✅ Voice Recognition Ready")
        st.info(f"👤 User ID: {st.session_state.user_id}")
    else:
        st.error("❌ ML System Unavailable")
    
    # Quick actions
    st.subheader("🚀 Quick Actions")
    if st.button("🔄 Refresh Recommendations"):
        st.session_state.recommendations = None
        st.rerun()
    
    if st.button("🗑️ Clear History"):
        st.session_state.user_id = f"user_{int(time.time())}"
        st.session_state.recommendations = None
        st.session_state.voice_transcription = ""
        if hasattr(st.session_state, 'voice_recommendations_data'):
            delattr(st.session_state, 'voice_recommendations_data')
        st.success("History cleared!")

    
    if st.button("📊 Export Results"):
        if st.session_state.recommendations:
            # Convert recommendations to CSV
            csv = st.session_state.recommendations['recommendations'].to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>🤖 Powered by Advanced ML Recommendation Engine | 
        🌤️ Real-time Weather Integration | 
        😊 Multimodal Sentiment Analysis</p>
    </div>
    """, 
    unsafe_allow_html=True
)

# Auto-refresh option
if st.sidebar.checkbox("🔄 Auto-refresh (Demo)", help="Refresh recommendations every 30 seconds"):
    time.sleep(30)
    st.rerun()


# Handle voice transcription messages from JavaScript
def handle_voice_messages():
    """Handle voice transcription messages from JavaScript components"""
    # This is a placeholder for the component communication
    # Streamlit components can communicate through session state updates
    # The HTML5 component updates will be handled through the component's return value
    pass