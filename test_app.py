import streamlit as st
import numpy as np
import pandas as pd

# Minimal test app for deployment
st.set_page_config(
    page_title="🌿 PlantMedic AI - Test",
    page_icon="🌿",
    layout="wide"
)

st.title("🌿 PlantMedic AI - Deployment Test")
st.write("Testing Streamlit Cloud deployment...")

# Test basic functionality
st.subheader("🧪 Testing Basic Dependencies")

try:
    import numpy as np
    st.success(f"✅ NumPy loaded successfully: {np.__version__}")
except Exception as e:
    st.error(f"❌ NumPy failed: {str(e)}")

try:
    import pandas as pd
    st.success(f"✅ Pandas loaded successfully: {pd.__version__}")
except Exception as e:
    st.error(f"❌ Pandas failed: {str(e)}")

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

st.subheader("📁 Testing File System")
try:
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    st.write(f"Script directory: {script_dir}")
    
    # List files in directory
    files = os.listdir(script_dir)
    st.write("Files in directory:")
    for file in files:
        st.write(f"- {file}")
        
except Exception as e:
    st.error(f"❌ File system test failed: {str(e)}")

st.subheader("🔬 TensorFlow Test (Optional)")
st.write("Now testing TensorFlow installation...")

try:
    import tensorflow as tf
    st.success(f"✅ TensorFlow loaded successfully: {tf.__version__}")
    tf_available = True
    
    # Test if model files exist
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "plant_disease_model.h5")
    class_names_path = os.path.join(script_dir, "class_names.pkl")
    
    st.write(f"Model exists: {os.path.exists(model_path)}")
    st.write(f"Class names exist: {os.path.exists(class_names_path)}")
    
    if os.path.exists(model_path) and os.path.exists(class_names_path):
        st.success("✅ Model files found! Ready for full app deployment.")
    else:
        st.warning("⚠️ Model files not found")
        
except ImportError as e:
    st.warning(f"⚠️ TensorFlow not available: {str(e)}")
    st.info("This is expected if TensorFlow was removed from requirements.txt")
    tf_available = False
except Exception as e:
    st.error(f"❌ TensorFlow failed: {str(e)}")
    tf_available = False

st.subheader("💡 Summary")
st.info("✅ Basic Streamlit deployment is working! If TensorFlow fails, we'll add it back separately.")
