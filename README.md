# � PlantMedic AI

Smart Plant Disease Detection & Agricultural Intelligence Platform

<p align="center">
  <img src="https://img.shields.io/badge/Made%20with-Streamlit-ff4b4b?logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Model-TensorFlow-orange?logo=tensorflow" />
  <img src="https://img.shields.io/badge/Dataset-PlantVillage-blueviolet?logo=kaggle" />
  <img src="https://img.shields.io/badge/License-MIT-green" />
  <img src="https://img.shields.io/badge/Status-Active-success" />
  <img src="https://img.shields.io/badge/Maintained%20by-Parthiv%20Koli-blue" />
  <img src="https://img.shields.io/badge/Version-2.0-brightgreen" />
  <img src="https://img.shields.io/badge/AI-Plant%20Health-success" />
</p>

---

## 🚀 Overview

**PlantMedic AI** is an enhanced, user-friendly AI-powered agricultural intelligence platform designed to help farmers, gardeners, and researchers diagnose plant diseases from leaf images. Built with Streamlit and TensorFlow, the app leverages a fine-tuned MobileNetV2 model trained on the [PlantVillage dataset](https://www.kaggle.com/datasets/emmarex/plantdisease) to detect diseases in tomato plants and provide comprehensive care recommendations.

### ✨ New Features in v2.0:
- **📊 Analytics Dashboard**: Track diagnosis history and patterns
- **🎨 Enhanced UI**: Modern design with animations and better UX
- **📝 Diagnosis History**: Keep track of all your plant diagnoses
- **📥 Export Capabilities**: Download diagnosis history as CSV
- **⚡ Performance Monitoring**: Backend optimizations for faster processing
- **🔧 Modular Architecture**: Better code organization and maintainability
- **🌿 New Branding**: Professional PlantMedic AI identity

---

## 🧠 How It Works

1. **Upload** a clear image of a plant leaf through the intuitive interface.
2. The **enhanced AI model** analyzes the image with improved preprocessing.
3. The app displays comprehensive results including:
   - The **predicted disease** with severity level
   - **Confidence score** with visual indicators
   - **Detailed description** and **symptoms**
   - **Treatment recommendations** and **prevention tips**
   - **Timeline** for treatment and **economic impact**

### 🎯 New Analysis Features:
- **Multi-tab interface** for different functionalities
- **Real-time analytics** of your diagnosis patterns
- **Enhanced disease information** with symptoms and prevention
- **Confidence visualization** and top predictions display

---

## 🖥️ Features

- 📸 Image upload and real-time diagnosis
- 🧪 AI model trained on the PlantVillage dataset
- ✅ Disease-specific descriptions and remedies
- 🌗 Responsive light/dark mode styling
- 🎈 Friendly UI with progress animation and balloon success celebration
- 📎 Linked GitHub repo and contact info
- 🚀 **Live Demo:** [ai-agrovision.streamlit.app](https://ai-agrovision.streamlit.app/) ([Preview](https://ai-agrovision.streamlit.app/))

---

## 🧰 Tech Stack

| Component | Technology                   |
|-----------|------------------------------|
| Frontend  | Streamlit                    |
| Model     | TensorFlow (MobileNetV2)     |
| Dataset   | [PlantVillage (Tomato subset)](https://www.kaggle.com/datasets/emmarex/plantdisease) |
| Styling   | Custom HTML/CSS              |
| Deployment| Localhost / Streamlit Cloud  |

---

## 📂 Project Structure

```
AgroVision/
├── app.py                       # Main Streamlit app
├── model/
│   ├── plant_disease_model.h5   # Trained TensorFlow model
│   └── class_names.pkl          # Class labels for predictions
├── data/
│   └── plantvillage/            # PlantVillage dataset (download from Kaggle)
├── assets/                      # (optional) images, logos, etc.
└── README.md
```

---

## 📸 Supported Tomato Classes

Example classes detected by the model:

- `Tomato___Bacterial_spot`
- `Tomato___Early_blight`
- `Tomato___healthy`
- _(More can be added as the model expands)_

---

## 🔧 Setup Instructions

1. **Clone the repository**
    ```bash
    git clone https://github.com/Parthivkoli/AgroVision.git
    cd AgroVision
    ```

2. **Download the PlantVillage dataset**  
   Get it from [Kaggle - PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease) and place it in the `data/plantvillage/` directory.

3. **Install required libraries**
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the app**
    ```bash
    streamlit run app.py
    ```

5. **Visit the app**  
   Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📦 Requirements

Include the following in `requirements.txt`:
```
streamlit
tensorflow
pillow
numpy
scikit-learn
```

---

## 🧑‍💻 Author

- **Parthiv Koli**
- 📧 parthivkoli69@gmail.com
- 🔗 [GitHub Profile](https://github.com/Parthivkoli)

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## ⭐ Acknowledgments

- [PlantVillage Dataset on Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease)
- [Streamlit Community](https://discuss.streamlit.io/)
- [TensorFlow Documentation](https://www.tensorflow.org/)

---

## 🙌 Contribute

Feel free to fork, improve, and submit a pull request. Contributions and feedback are welcome!
