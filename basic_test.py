import streamlit as st

st.title("🌿 PlantMedic AI - Basic Test")
st.write("If you can see this, Streamlit is working!")

st.subheader("Success!")
st.success("✅ Streamlit Cloud deployment is working!")

st.info("Next step: Add back dependencies one by one.")

# Test built-in Python modules
import os
import sys

st.write(f"Python version: {sys.version}")
st.write(f"Current directory: {os.getcwd()}")

# List files
files = os.listdir('.')
st.write("Files in directory:")
for file in files:
    st.write(f"- {file}")
