import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

# Test app for deployment
st.set_page_config(
    page_title="🌿 PlantMedic AI - Test",
    page_icon="🌿",
    layout="wide"
)

st.title("🌿 PlantMedic AI - Deployment Test")
st.write("Testing Streamlit Cloud deployment...")

# Test basic functionality
st.subheader("🧪 Testing Dependencies")

try:
    import tensorflow as tf
    st.success(f"✅ TensorFlow loaded successfully: {tf.__version__}")
    tf_available = True
except Exception as e:
    st.error(f"❌ TensorFlow failed: {str(e)}")
    tf_available = False

try:
    from PIL import Image
    st.success("✅ PIL/Pillow loaded successfully")
except Exception as e:
    st.error(f"❌ PIL failed: {str(e)}")

try:
    import plotly.express as px
    st.success("✅ Plotly loaded successfully")
    
    # Test plot
    df = pd.DataFrame({
        'x': [1, 2, 3, 4],
        'y': [10, 11, 12, 13]
    })
    fig = px.line(df, x='x', y='y', title='Test Plot')
    st.plotly_chart(fig)
except Exception as e:
    st.error(f"❌ Plotly failed: {str(e)}")

if tf_available:
    st.subheader("🔬 Testing Model Loading")
    try:
        # Test if model files exist
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, "plant_disease_model.h5")
        class_names_path = os.path.join(script_dir, "class_names.pkl")
        
        st.write(f"Script directory: {script_dir}")
        st.write(f"Model exists: {os.path.exists(model_path)}")
        st.write(f"Class names exist: {os.path.exists(class_names_path)}")
        
        if os.path.exists(model_path) and os.path.exists(class_names_path):
            st.success("✅ Model files found!")
        else:
            st.warning("⚠️ Model files not found")
            
    except Exception as e:
        st.error(f"❌ Model loading test failed: {str(e)}")

st.subheader("💡 Next Steps")
if tf_available:
    st.info("🎉 All dependencies loaded successfully! Ready to enable full app.")
else:
    st.warning("⚠️ TensorFlow issues detected. Check requirements.txt compatibility.")
