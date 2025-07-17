import streamlit as st
from PIL import Ima    # Enhanced configuration with new branding
    APP_TITLE = "🌿 PlantMedic AI"
    APP_SUBTITLE = "Smart Plant Disease Detection & Agricultural Intelligence Platform"
    APP_LOGO = "https://cdn-icons-png.flaticon.com/512/628/628283.png"  # Modern plant health icon
    GITHUB_LINK = "https://github.com/Parthivkoli/AgroVision"
    MODEL_PATH = "plant_disease_model.h5"  # Try root first, fallback to model/
    CLASS_NAMES_PATH = "class_names.pkl"  # Try root first, fallback to model/ort numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import pickle
import time
import logging
import json
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Import custom modules
try:
    from config import config, ENHANCED_DISEASE_INFO
    from utils import (
        image_processor, model_manager, data_manager, 
        performance_monitor, cache_manager
    )
    USE_ENHANCED_FEATURES = True
except ImportError:
    # Fallback to basic functionality if custom modules not found
    USE_ENHANCED_FEATURES = False
    logging.warning("Custom modules not found, using basic functionality")

# --- Customization with Config Support ---
if USE_ENHANCED_FEATURES:
    APP_TITLE = config.ui.app_title
    APP_SUBTITLE = config.ui.app_subtitle
    APP_LOGO = config.ui.app_logo
    GITHUB_LINK = config.ui.github_link
    MODEL_PATH = config.model.model_path
    CLASS_NAMES_PATH = config.model.class_names_path
    ENABLE_HISTORY = config.features.enable_history
    ENABLE_ANALYTICS = config.features.enable_analytics
    ENABLE_EXPORT = config.features.enable_export
    DISEASE_INFO = ENHANCED_DISEASE_INFO
else:
    # Enhanced configuration with new branding
    APP_TITLE = "� PlantMedic AI"
    APP_SUBTITLE = "Smart Plant Disease Detection & Agricultural Intelligence Platform"
    APP_LOGO = "https://cdn-icons-png.flaticon.com/512/628/628283.png"  # Modern plant health icon
    GITHUB_LINK = "https://github.com/Parthivkoli/AgroVision"
    MODEL_PATH = "plant_disease_model.h5"
    CLASS_NAMES_PATH = "class_names.pkl"
    ENABLE_HISTORY = True
    ENABLE_ANALYTICS = True
    ENABLE_EXPORT = True
    DISEASE_INFO = {
        "Tomato___healthy": {
            "desc": "Your plant is healthy and thriving!",
            "remedy": "Continue good care practices.",
            "severity": "None",
            "prevention": "Maintain current care routine.",
            "symptoms": ["Vibrant green leaves"]
        }
    }

# Disease information dictionary with enhanced data
DISEASE_INFO = {
    "Tomato___Bacterial_spot": {
        "desc": "Bacterial spot is a common disease in tomatoes caused by Xanthomonas species. It affects leaves, stems, and fruits.",
        "remedy": "Remove infected leaves, avoid overhead watering, and apply copper-based fungicides. Ensure proper spacing for air circulation.",
        "severity": "High",
        "prevention": "Use disease-free seeds, practice crop rotation, and maintain proper plant spacing.",
        "symptoms": ["Small dark spots on leaves", "Yellow halos around spots", "Fruit lesions"]
    },
    "Tomato___Early_blight": {
        "desc": "Early blight, caused by Alternaria solani, leads to dark spots with concentric rings on leaves and fruit.",
        "remedy": "Remove affected leaves, practice crop rotation, and apply fungicides like chlorothalonil or mancozeb.",
        "severity": "Medium",
        "prevention": "Ensure good air circulation, avoid overhead watering, and remove plant debris.",
        "symptoms": ["Concentric ring patterns", "Brown to black spots", "Yellowing leaves"]
    },
    "Tomato___healthy": {
        "desc": "Your plant is healthy and thriving! No signs of disease detected.",
        "remedy": "Continue good care practices: proper watering, balanced fertilization, and regular monitoring.",
        "severity": "None",
        "prevention": "Maintain current care routine and monitor regularly.",
        "symptoms": ["Vibrant green leaves", "No spots or discoloration", "Healthy growth"]
    },
    # Add more entries as needed
}

# --- Set page configuration ---
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for history and analytics
if 'diagnosis_history' not in st.session_state:
    st.session_state.diagnosis_history = []
if 'analytics_data' not in st.session_state:
    st.session_state.analytics_data = {
        'total_diagnoses': 0,
        'disease_counts': {},
        'confidence_scores': []
    }

# --- Custom CSS for enhanced theme-aware styling ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    :root {
        --primary-bg: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --secondary-bg: #f8f9ff;
        --sidebar-bg: #ffffff;
        --header-bg: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        --text-color: #2d3748;
        --accent-color: #4299e1;
        --success-color: #48bb78;
        --warning-color: #ed8936;
        --error-color: #f56565;
        --border-radius: 12px;
        --box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }

    * {
        font-family: 'Inter', sans-serif !important;
    }

    .stApp {
        background: var(--secondary-bg);
    }

    .main-header {
        background: var(--header-bg);
        padding: 2rem;
        border-radius: var(--border-radius);
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: var(--box-shadow);
        animation: fadeInDown 0.8s ease-out;
    }

    .main-header h1 {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }

    .main-header h4 {
        font-size: 1.2rem;
        font-weight: 400;
        opacity: 0.9;
    }

    .upload-section {
        background: white;
        padding: 2rem;
        border-radius: var(--border-radius);
        box-shadow: var(--box-shadow);
        margin-bottom: 2rem;
        border: 2px dashed #e2e8f0;
        transition: all 0.3s ease;
    }

    .upload-section:hover {
        border-color: var(--accent-color);
        transform: translateY(-2px);
    }

    .results-card {
        background: white;
        padding: 2rem;
        border-radius: var(--border-radius);
        box-shadow: var(--box-shadow);
        margin: 1rem 0;
        border-left: 4px solid var(--success-color);
        animation: fadeInUp 0.6s ease-out;
    }

    .severity-high { border-left-color: var(--error-color) !important; }
    .severity-medium { border-left-color: var(--warning-color) !important; }
    .severity-low { border-left-color: var(--success-color) !important; }

    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: var(--border-radius);
        text-align: center;
        box-shadow: var(--box-shadow);
        transition: transform 0.3s ease;
    }

    .metric-card:hover {
        transform: translateY(-5px);
    }

    .sidebar-section {
        background: white;
        padding: 1.5rem;
        border-radius: var(--border-radius);
        margin-bottom: 1rem;
        box-shadow: var(--box-shadow);
    }

    .feature-button {
        background: var(--accent-color);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: var(--border-radius);
        cursor: pointer;
        transition: all 0.3s ease;
        width: 100%;
        margin: 0.25rem 0;
    }

    .feature-button:hover {
        background: #3182ce;
        transform: translateY(-2px);
    }

    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .diagnosis-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 0.9rem;
        margin: 0.25rem;
    }

    .badge-healthy { background: #c6f6d5; color: #22543d; }
    .badge-diseased { background: #fed7d7; color: #742a2a; }
    .badge-warning { background: #fef5e7; color: #744210; }

    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
        :root {
            --secondary-bg: #1a202c;
            --sidebar-bg: #2d3748;
            --text-color: #e2e8f0;
        }
        
        .upload-section, .results-card, .sidebar-section {
            background: #2d3748;
            color: var(--text-color);
        }
    }

    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: var(--border-radius);
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: var(--box-shadow);
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.2);
    }

    .stSelectbox>div>div {
        border-radius: var(--border-radius);
        border: 2px solid #e2e8f0;
    }

    .stSelectbox>div>div:focus-within {
        border-color: var(--accent-color);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Enhanced Sidebar ---
with st.sidebar:
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.image(APP_LOGO, width=120)
    st.title(APP_TITLE)
    st.markdown(f"_{APP_SUBTITLE}_")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Navigation tabs
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("### 🚀 Features")
    tab_option = st.selectbox(
        "Choose a feature:",
        ["🔍 Diagnosis", "📊 Analytics", "📝 History", "ℹ️ About"],
        index=0
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Model information
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("### 🤖 Model Info")
    st.metric("Model", "MobileNetV2")
    st.metric("Dataset", "PlantVillage")
    st.metric("Accuracy", "94.2%")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Quick stats
    if ENABLE_ANALYTICS and st.session_state.analytics_data['total_diagnoses'] > 0:
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("### 📈 Quick Stats")
        st.metric("Total Diagnoses", st.session_state.analytics_data['total_diagnoses'])
        if st.session_state.analytics_data['confidence_scores']:
            avg_confidence = np.mean(st.session_state.analytics_data['confidence_scores'])
            st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Contact and source
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("### 📞 Contact & Source")
    st.markdown(f"[🔗 GitHub Repository]({GITHUB_LINK})")
    st.markdown("**Developer:** [Parthiv Koli](mailto:parthivkoli@example.com)")
    st.markdown("Built with ❤️ using Streamlit & TensorFlow")
    st.markdown('</div>', unsafe_allow_html=True)

# --- Main content with enhanced header ---
st.markdown(
    f"""
    <div class="main-header">
        <h1>🌱 {APP_TITLE}</h1>
        <h4>{APP_SUBTITLE}</h4>
        <p style="margin-top: 1rem; opacity: 0.9;">Powered by Advanced Machine Learning • Real-time Disease Detection • Expert Care Recommendations</p>
    </div>
    """,
    unsafe_allow_html=True
)

# --- Load model and class names ---
@st.cache_resource
def load_model_and_classes():
    try:
        # Load class names to determine number of classes
        with open(CLASS_NAMES_PATH, 'rb') as f:
            class_names = pickle.load(f)
        num_classes = len(class_names)
        
        # Create MobileNetV2 base model
        base_model = MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights=None  # We’ll load custom weights
        )
        
        # Add custom layers to match the saved model architecture
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation='relu')(x)  # Changed from 1024 to 128 to match weights
        predictions = Dense(num_classes, activation='softmax')(x)
        
        # Create the full model
        model = Model(inputs=base_model.input, outputs=predictions)
        
        # Load the weights from the .h5 file
        model.load_weights(MODEL_PATH)
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model, class_names
    except Exception as e:
        logging.error(f"Error loading model or class names: {str(e)}")
        st.error(f"""
        Failed to load the model or class names. Error details:
        {str(e)}
        
        Please ensure:
        1. The model weights file exists at {MODEL_PATH}
        2. The class names file exists at {CLASS_NAMES_PATH}
        3. The model weights are compatible with TensorFlow 2.12.0 and the defined architecture
        4. The class_names.pkl file is a valid pickle file containing a list of class names
        """)
        return None, None

model, class_names = load_model_and_classes()

if model is None or class_names is None:
    st.stop()

# --- Helper Functions for New Features ---
def save_diagnosis_to_history(image, prediction, confidence, timestamp):
    """Save diagnosis to session state history"""
    if ENABLE_HISTORY:
        history_entry = {
            'timestamp': timestamp,
            'prediction': prediction,
            'confidence': confidence,
            'image_name': getattr(uploaded_file, 'name', 'Unknown')
        }
        st.session_state.diagnosis_history.append(history_entry)
        
        # Update analytics
        st.session_state.analytics_data['total_diagnoses'] += 1
        st.session_state.analytics_data['confidence_scores'].append(confidence * 100)
        
        disease_name = prediction.replace('_', ' ')
        if disease_name in st.session_state.analytics_data['disease_counts']:
            st.session_state.analytics_data['disease_counts'][disease_name] += 1
        else:
            st.session_state.analytics_data['disease_counts'][disease_name] = 1

def get_severity_class(severity):
    """Get CSS class for severity level"""
    if severity == "High":
        return "severity-high"
    elif severity == "Medium":
        return "severity-medium"
    else:
        return "severity-low"

def render_diagnosis_tab():
    """Render the main diagnosis interface"""
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 📷 Upload Plant Image")
        uploaded_file = st.file_uploader(
            "Choose a plant leaf image",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear, well-lit image of a plant leaf for best results.",
            key="image_uploader"
        )
        
        if uploaded_file:
            st.markdown("**Tips for better results:**")
            st.markdown("• Use good lighting")
            st.markdown("• Focus on the affected area")
            st.markdown("• Avoid blurry images")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    return uploaded_file, col2

def render_analytics_tab():
    """Render analytics dashboard"""
    st.markdown("### 📊 Analytics Dashboard")
    
    if st.session_state.analytics_data['total_diagnoses'] == 0:
        st.info("No data available yet. Start by diagnosing some plants!")
        return
    
    # Overview metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Diagnoses", st.session_state.analytics_data['total_diagnoses'])
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        avg_conf = np.mean(st.session_state.analytics_data['confidence_scores'])
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Average Confidence", f"{avg_conf:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        unique_diseases = len(st.session_state.analytics_data['disease_counts'])
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Unique Conditions", unique_diseases)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Disease distribution pie chart
        if st.session_state.analytics_data['disease_counts']:
            fig_pie = px.pie(
                values=list(st.session_state.analytics_data['disease_counts'].values()),
                names=list(st.session_state.analytics_data['disease_counts'].keys()),
                title="Disease Distribution"
            )
            fig_pie.update_layout(
                font=dict(family="Inter", size=12),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Confidence scores histogram
        if st.session_state.analytics_data['confidence_scores']:
            fig_hist = px.histogram(
                x=st.session_state.analytics_data['confidence_scores'],
                title="Confidence Score Distribution",
                labels={'x': 'Confidence (%)', 'y': 'Count'},
                nbins=20
            )
            fig_hist.update_layout(
                font=dict(family="Inter", size=12),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_hist, use_container_width=True)

def render_history_tab():
    """Render diagnosis history"""
    st.markdown("### 📝 Diagnosis History")
    
    if not st.session_state.diagnosis_history:
        st.info("No diagnosis history available yet.")
        return
    
    # Create DataFrame for better display
    history_df = pd.DataFrame(st.session_state.diagnosis_history)
    history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
    history_df = history_df.sort_values('timestamp', ascending=False)
    
    # Display recent diagnoses
    st.markdown("#### Recent Diagnoses")
    for idx, row in history_df.head(10).iterrows():
        with st.expander(f"🔍 {row['prediction'].replace('_', ' ')} - {row['timestamp'].strftime('%Y-%m-%d %H:%M')}"):
            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric("Confidence", f"{row['confidence']*100:.1f}%")
                st.text(f"Image: {row['image_name']}")
            with col2:
                info = DISEASE_INFO.get(row['prediction'], {})
                if info:
                    st.write(f"**Description:** {info.get('desc', 'N/A')}")
                    st.write(f"**Severity:** {info.get('severity', 'Unknown')}")
    
    # Export option
    if st.button("📥 Export History as CSV"):
        csv = history_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"agrovision_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

def render_about_tab():
    """Render about information"""
    st.markdown("### ℹ️ About AgroVision Pro")
    
    st.markdown("""
    **PlantMedic AI** is an advanced AI-powered plant disease detection system that helps farmers, 
    gardeners, and agricultural professionals identify plant diseases quickly and accurately.
    
    #### 🎯 Key Features:
    - **Real-time Disease Detection**: Upload an image and get instant results
    - **Expert Recommendations**: Detailed treatment and prevention advice
    - **Analytics Dashboard**: Track your diagnosis history and patterns
    - **Export Capabilities**: Download your diagnosis history for record keeping
    
    #### 🤖 Technology Stack:
    - **Frontend**: Streamlit with custom CSS and animations
    - **Backend**: TensorFlow 2.12 with MobileNetV2 architecture
    - **Dataset**: PlantVillage dataset with 38+ plant disease classes
    - **Accuracy**: 94.2% validation accuracy
    
    #### 🌱 Supported Plants:
    Currently supports tomato plants with plans to expand to more crops.
    
    #### 📞 Support:
    For technical support or feature requests, please visit our [GitHub repository]({}) or contact the developer.
    """.format(GITHUB_LINK))

# --- Tab Content Rendering ---
if tab_option == "🔍 Diagnosis":
    uploaded_file, col2 = render_diagnosis_tab()
elif tab_option == "📊 Analytics":
    render_analytics_tab()
    uploaded_file = None
elif tab_option == "📝 History":
    render_history_tab()
    uploaded_file = None
elif tab_option == "ℹ️ About":
    render_about_tab()
    uploaded_file = None
else:
    uploaded_file, col2 = render_diagnosis_tab()

# --- Enhanced Image Processing ---
if uploaded_file is not None and tab_option == "🔍 Diagnosis":
    try:
        image = Image.open(uploaded_file).convert('RGB')
        
        # Enhanced preprocessing
        img = image.resize((224, 224))
        img_array = (np.array(img) / 255.0).astype(np.float32)
        img_array = np.expand_dims(img_array, axis=0)

        # Animated progress bar
        progress_container = st.container()
        with progress_container:
            st.markdown("### 🔄 Analyzing Image...")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            analysis_steps = [
                "Loading image...",
                "Preprocessing...", 
                "Running AI analysis...",
                "Calculating confidence...",
                "Generating recommendations..."
            ]
            
            for i, step in enumerate(analysis_steps):
                time.sleep(0.3)
                progress = (i + 1) / len(analysis_steps)
                progress_bar.progress(progress)
                status_text.text(step)
        
        # Make prediction
        prediction = model.predict(img_array)
        pred_idx = np.argmax(prediction[0])
        pred_class = class_names[pred_idx]
        pred_prob = prediction[0][pred_idx]
        
        # Clear progress indicators
        progress_container.empty()
        
        # Save to history
        current_time = datetime.now()
        save_diagnosis_to_history(image, pred_class, pred_prob, current_time)

        # --- Enhanced Results Display ---
        with col2:
            # Image display
            st.markdown("### 📸 Uploaded Image")
            st.image(image, caption='Plant Leaf for Analysis', width=300, clamp=True)
            
        # Results card
        info = DISEASE_INFO.get(pred_class, {})
        severity = info.get('severity', 'Unknown')
        severity_class = get_severity_class(severity)
        
        st.markdown(f'<div class="results-card {severity_class}">', unsafe_allow_html=True)
        
        # Main results
        col_result1, col_result2, col_result3 = st.columns(3)
        
        with col_result1:
            st.markdown("### 🎯 Diagnosis")
            disease_name = pred_class.replace('_', ' ').title()
            st.markdown(f"**{disease_name}**")
            
            # Severity badge
            if severity == "High":
                st.markdown('<span class="diagnosis-badge badge-diseased">🔴 High Severity</span>', unsafe_allow_html=True)
            elif severity == "Medium":
                st.markdown('<span class="diagnosis-badge badge-warning">🟡 Medium Severity</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="diagnosis-badge badge-healthy">🟢 Healthy/Low Risk</span>', unsafe_allow_html=True)
        
        with col_result2:
            st.markdown("### 📊 Confidence")
            st.metric("Accuracy Score", f"{pred_prob*100:.1f}%")
            
            # Confidence indicator
            if pred_prob > 0.9:
                st.success("Very High Confidence")
            elif pred_prob > 0.7:
                st.info("High Confidence") 
            else:
                st.warning("Moderate Confidence")
        
        with col_result3:
            st.markdown("### ⏰ Analysis Time")
            st.metric("Processed", current_time.strftime("%H:%M:%S"))
            st.caption(current_time.strftime("%Y-%m-%d"))
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Detailed Information Sections
        if info:
            # Description
            st.markdown('<div class="results-card">', unsafe_allow_html=True)
            st.markdown("### 📋 Description")
            st.info(info.get('desc', 'No description available.'))
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Symptoms
            if 'symptoms' in info:
                st.markdown('<div class="results-card">', unsafe_allow_html=True)
                st.markdown("### 🔍 Key Symptoms")
                for symptom in info['symptoms']:
                    st.markdown(f"• {symptom}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Treatment
            st.markdown('<div class="results-card">', unsafe_allow_html=True)
            st.markdown("### 💊 Recommended Treatment")
            st.success(info.get('remedy', 'No specific treatment available.'))
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Prevention
            if 'prevention' in info:
                st.markdown('<div class="results-card">', unsafe_allow_html=True)
                st.markdown("### 🛡️ Prevention Tips")
                st.warning(info.get('prevention', 'No prevention tips available.'))
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Top predictions
        st.markdown('<div class="results-card">', unsafe_allow_html=True)
        st.markdown("### 📈 Top Predictions")
        
        # Get top 3 predictions
        top_indices = np.argsort(prediction[0])[-3:][::-1]
        
        for i, idx in enumerate(top_indices):
            disease = class_names[idx].replace('_', ' ').title()
            confidence = prediction[0][idx] * 100
            
            col_pred1, col_pred2 = st.columns([3, 1])
            with col_pred1:
                st.write(f"{i+1}. {disease}")
            with col_pred2:
                st.write(f"{confidence:.1f}%")
            
            # Progress bar for confidence
            st.progress(confidence / 100)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Success animation
        st.balloons()
        
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        st.error("❌ An error occurred while processing the image. Please try uploading a different image.")
        
elif uploaded_file is None and tab_option == "🔍 Diagnosis":
    # Welcome message with tips
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.markdown("### 🌟 Welcome to PlantMedic AI!")
    st.info("👆 Upload a plant leaf image to begin AI-powered disease diagnosis!")
    
    st.markdown("#### 💡 Tips for Best Results:")
    st.markdown("""
    - **Lighting**: Use natural light or bright, even lighting
    - **Focus**: Ensure the leaf fills most of the frame
    - **Clarity**: Avoid blurry or low-resolution images
    - **Background**: Plain backgrounds work best
    - **Angle**: Take photos straight-on when possible
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# --- Enhanced Footer ---
st.markdown("---")
st.markdown(
    f"""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                border-radius: 12px; color: white; margin-top: 3rem;">
        <h3>� PlantMedic AI</h3>
        <p>Empowering Agriculture with AI • Making Plant Health Accessible to Everyone</p>
        <p style="margin-top: 1rem;">
            <a href="{GITHUB_LINK}" target="_blank" style="color: white; text-decoration: none;">
                🔗 Open Source on GitHub
            </a> | 
            <a href="mailto:parthivkoli@example.com" style="color: white; text-decoration: none;">
                📧 Contact Developer
            </a>
        </p>
        <p style="font-size: 0.9rem; opacity: 0.8; margin-top: 1rem;">
            Built with ❤️ using Streamlit, TensorFlow & Modern Web Technologies
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
