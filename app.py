import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import pickle
import time
import logging

# --- Customization ---
APP_TITLE = "🌱 AgroVision"
APP_SUBTITLE = "Diagnose Plant Diseases with AI Precision"
APP_LOGO = "https://img.icons8.com/fluency/96/plant-under-sun.png"
GITHUB_LINK = "https://github.com/Parthivkoli/AgroVision"
MODEL_PATH = "model/plant_disease_model.h5"
CLASS_NAMES_PATH = "model/class_names.pkl"

# Disease information dictionary
DISEASE_INFO = {
    "Tomato___Bacterial_spot": {
        "desc": "Bacterial spot is a common disease in tomatoes caused by Xanthomonas species.",
        "remedy": "Remove infected leaves, avoid overhead watering, and apply copper-based fungicides. Ensure proper spacing for air circulation."
    },
    "Tomato___Early_blight": {
        "desc": "Early blight, caused by Alternaria solani, leads to dark spots with concentric rings on leaves and fruit.",
        "remedy": "Remove affected leaves, practice crop rotation, and apply fungicides like chlorothalonil or mancozeb."
    },
    "Tomato___healthy": {
        "desc": "Your plant is healthy and thriving!",
        "remedy": "Continue good care practices: proper watering, balanced fertilization, and regular monitoring."
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

# --- Custom CSS for theme-aware styling ---
st.markdown(
    """
    <style>
    :root {
        --primary-bg: #f0f4f8;
        --sidebar-bg: #e8f5e9;
        --header-bg: #ffffff;
        --text-color: #333333;
        --footer-color: #666666;
        --button-bg: #4CAF50;
        --button-hover-bg: #45a049;
        --alert-bg: #e0e0e0;
    }

    /* Dark mode adjustments */
    [data-testid="stAppViewContainer"] {
        background-color: var(--primary-bg);
    }
    [data-testid="stSidebar"] {
        background-color: var(--sidebar-bg);
        border-right: 1px solid #d0d0d0;
    }

    /* Theme detection for dark mode */
    @media (prefers-color-scheme: dark) {
        :root {
            --primary-bg: #1f2a44;
            --sidebar-bg: #2a3b57;
            --header-bg: #2e3b55;
            --text-color: #e0e0e0;
            --footer-color: #a0a0a0;
            --alert-bg: #3a4a6b;
        }
        [data-testid="stAppViewContainer"] {
            background-color: var(--primary-bg);
        }
        [data-testid="stSidebar"] {
            background-color: var(--sidebar-bg);
        }
        .header, .footer, .stMarkdown, .stAlert, .stMetricLabel, .stMetricValue, .stCaption {
            color: var(--text-color) !important;
        }
        .stAlert {
            background-color: var(--alert-bg);
        }
    }

    .stButton>button {
        background-color: var(--button-bg);
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: var(--button-hover-bg);
    }
    .stAlert {
        border-radius: 10px;
        padding: 15px;
        color: var(--text-color);
        background-color: var(--alert-bg);
    }
    .header {
        text-align: center;
        padding: 20px;
        background-color: var(--header-bg);
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        color: var(--text-color);
    }
    .footer {
        text-align: center;
        color: var(--footer-color);
        padding: 20px;
        margin-top: 20px;
        font-size: 14px;
    }
    .stMarkdown, .stCaption {
        color: var(--text-color);
    }
    .image-container {
        display: flex;
        align-items: flex-start;
        gap: 20px;
    }
    .diagnosis-results {
        flex: 1;
    }
    .uploaded-image {
        max-width: 200px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Sidebar ---
with st.sidebar:
    st.image(APP_LOGO, width=120)
    st.title(APP_TITLE)
    st.markdown(f"_{APP_SUBTITLE}_")
    st.markdown("---")
    st.markdown(
        """
        **How it works:**
        Upload a clear image of a plant leaf, and our AI will analyze it to detect potential diseases and provide tailored remedies.
        
        **Model:** MobileNetV2, fine-tuned on PlantVillage dataset  
        **Source:** [AgroVision on GitHub]({})
        """.format(GITHUB_LINK)
    )
    st.markdown("**Contact:** [Parthiv Koli](mailto:parthivkoli@example.com)")
    st.markdown("---")
    st.caption("Built with Streamlit and TensorFlow")

# --- Main content ---
st.markdown(
    f"""
    <div class="header">
        <h1>{APP_TITLE}</h1>
        <h4>{APP_SUBTITLE}</h4>
    </div>
    """,
    unsafe_allow_html=True
)

# --- Load model and class names ---
@st.cache_resource
def load_model_and_classes():
    try:
        # Define input layer explicitly
        inputs = tf.keras.Input(shape=(224, 224, 3))
        
        # Load model weights only
        base_model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False,
            custom_objects={
                'Input': lambda shape, **kwargs: tf.keras.layers.Input(shape=shape[-3:])
            }
        )
        
        # Create new model with explicit input
        model = tf.keras.Model(inputs=inputs, outputs=base_model(inputs))
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Load class names
        with open(CLASS_NAMES_PATH, 'rb') as f:
            class_names = pickle.load(f)
            
        return model, class_names
    except Exception as e:
        logging.error(f"Error loading model or class names: {str(e)}")
        st.error(f"""
        Failed to load the model or class names. Error details:
        {str(e)}
        
        Please ensure:
        1. The model file exists at {MODEL_PATH}
        2. The class names file exists at {CLASS_NAMES_PATH}
        3. You have TensorFlow 2.13.0 installed
        """)
        return None, None

model, class_names = load_model_and_classes()

if model is None or class_names is None:
    st.stop()

# --- Image uploader ---
col1, col2 = st.columns([1, 2])
with col1:
    uploaded_file = st.file_uploader(
        "Upload a plant leaf image (JPG, PNG)",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear image of a plant leaf for diagnosis."
    )

# --- Process uploaded image ---
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert('RGB')
        
        # Preprocess image
        img = image.resize((224, 224))
        img_array = (np.array(img) / 255.0).astype(np.float32)
        img_array = np.expand_dims(img_array, axis=0)

        # Progress bar for diagnosis
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)
        
        # Make prediction
        prediction = model.predict(img_array)
        pred_idx = np.argmax(prediction)
        pred_class = class_names[pred_idx]
        pred_prob = prediction[0][pred_idx]

        # --- Display image and results side by side ---
        with col2:
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            with st.container():
                st.image(image, caption='Uploaded Leaf Image', width=200, clamp=True)
            
            with st.container():
                st.markdown('<div class="diagnosis-results">', unsafe_allow_html=True)
                st.markdown("### Diagnosis Results")
                st.success(f"**Prediction:** {pred_class.replace('_', ' ')}")
                st.metric("Confidence", f"{pred_prob*100:.2f}%")
                
                info = DISEASE_INFO.get(pred_class, None)
                if info:
                    with st.expander("Learn More"):
                        st.info(f"**About:** {info['desc']}")
                        st.warning(f"**Suggested Remedy:** {info['remedy']}")
                else:
                    st.write("No additional information available for this diagnosis.")
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

        st.balloons()
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        st.error("An error occurred while processing the image. Please try uploading a different image.")
else:
    st.info("👆 Upload a plant leaf image to begin diagnosis!")

# --- Footer ---
st.markdown(
    f"""
    <div class="footer">
        Made with ❤️ using Streamlit | <a href="{GITHUB_LINK}" target="_blank">GitHub Repository</a>
    </div>
    """,
    unsafe_allow_html=True
)
