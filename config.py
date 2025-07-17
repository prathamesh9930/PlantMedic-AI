# AgroVision Pro Configuration File
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class ModelConfig:
    """Configuration for the ML model"""
    model_path: str = "plant_disease_model.h5"
    class_names_path: str = "class_names.pkl"
    input_shape: tuple = (224, 224, 3)
    confidence_threshold: float = 0.7
    top_predictions: int = 3

@dataclass
class UIConfig:
    """Configuration for UI settings"""
    app_title: str = "� PlantMedic AI"
    app_subtitle: str = "Smart Plant Disease Detection & Agricultural Intelligence Platform"
    app_logo: str = "https://cdn-icons-png.flaticon.com/512/628/628283.png"
    github_link: str = "https://github.com/Parthivkoli/AgroVision"
    page_icon: str = "🌿"
    layout: str = "wide"
    
@dataclass
class FeatureFlags:
    """Feature toggle configuration"""
    enable_history: bool = True
    enable_analytics: bool = True
    enable_export: bool = True
    enable_dark_mode: bool = True
    enable_animations: bool = True
    max_history_items: int = 100

@dataclass
class AnalyticsConfig:
    """Analytics configuration"""
    enable_performance_tracking: bool = True
    enable_user_feedback: bool = True
    save_analytics_locally: bool = True
    analytics_file: str = "analytics_data.json"

class AppConfig:
    """Main application configuration"""
    
    def __init__(self):
        self.model = ModelConfig()
        self.ui = UIConfig()
        self.features = FeatureFlags()
        self.analytics = AnalyticsConfig()
        
        # Environment-based overrides
        self._load_from_environment()
    
    def _load_from_environment(self):
        """Load configuration from environment variables"""
        # Model configuration
        if os.getenv('MODEL_PATH'):
            self.model.model_path = os.getenv('MODEL_PATH')
        if os.getenv('CLASS_NAMES_PATH'):
            self.model.class_names_path = os.getenv('CLASS_NAMES_PATH')
        if os.getenv('CONFIDENCE_THRESHOLD'):
            self.model.confidence_threshold = float(os.getenv('CONFIDENCE_THRESHOLD'))
        
        # UI configuration
        if os.getenv('APP_TITLE'):
            self.ui.app_title = os.getenv('APP_TITLE')
        if os.getenv('GITHUB_LINK'):
            self.ui.github_link = os.getenv('GITHUB_LINK')
        
        # Feature flags
        if os.getenv('ENABLE_ANALYTICS'):
            self.features.enable_analytics = os.getenv('ENABLE_ANALYTICS').lower() == 'true'
        if os.getenv('ENABLE_HISTORY'):
            self.features.enable_history = os.getenv('ENABLE_HISTORY').lower() == 'true'

# Global configuration instance
config = AppConfig()

# Enhanced disease information with more details
ENHANCED_DISEASE_INFO = {
    "Tomato___Bacterial_spot": {
        "desc": "Bacterial spot is a common disease in tomatoes caused by Xanthomonas species. It affects leaves, stems, and fruits, causing significant yield losses if left untreated.",
        "remedy": "Remove infected leaves immediately, avoid overhead watering, apply copper-based fungicides (copper sulfate or copper hydroxide). Ensure proper plant spacing for air circulation.",
        "severity": "High",
        "prevention": "Use disease-free seeds, practice 3-year crop rotation, maintain proper plant spacing, avoid working with wet plants, and use drip irrigation instead of overhead watering.",
        "symptoms": ["Small dark spots on leaves", "Yellow halos around spots", "Fruit lesions", "Defoliation in severe cases"],
        "treatment_timeline": "7-14 days with proper treatment",
        "economic_impact": "Can reduce yield by 20-50%",
        "favorable_conditions": "Warm, humid weather (75-86°F, high humidity)"
    },
    "Tomato___Early_blight": {
        "desc": "Early blight, caused by Alternaria solani, is a fungal disease that creates characteristic 'target spot' lesions on leaves and can affect fruit quality.",
        "remedy": "Remove affected leaves below the first fruit cluster, apply fungicides containing chlorothalonil, mancozeb, or azoxystrobin. Improve air circulation and reduce leaf wetness.",
        "severity": "Medium",
        "prevention": "Ensure good air circulation, avoid overhead watering, remove plant debris, practice crop rotation, and maintain adequate spacing between plants.",
        "symptoms": ["Concentric ring patterns (target spots)", "Brown to black spots on older leaves", "Yellowing and browning of leaves", "Stem lesions"],
        "treatment_timeline": "10-21 days with fungicide treatment",
        "economic_impact": "Moderate yield loss if untreated",
        "favorable_conditions": "Warm temperatures (75-85°F) with periods of leaf wetness"
    },
    "Tomato___Late_blight": {
        "desc": "Late blight is caused by Phytophthora infestans and is one of the most destructive tomato diseases. It can rapidly destroy entire crops under favorable conditions.",
        "remedy": "Apply preventive fungicides (copper-based or systemic), remove infected plants immediately, improve air circulation, and avoid overhead watering.",
        "severity": "High",
        "prevention": "Use resistant varieties, ensure proper spacing, avoid overhead irrigation, apply preventive fungicides during humid periods.",
        "symptoms": ["Water-soaked lesions on leaves", "White fuzzy growth on leaf undersides", "Brown-black lesions on stems and fruits", "Rapid plant death"],
        "treatment_timeline": "Act immediately - disease can destroy plants in days",
        "economic_impact": "Can cause total crop loss",
        "favorable_conditions": "Cool, wet weather (60-70°F with high humidity)"
    },
    "Tomato___Leaf_Mold": {
        "desc": "Leaf mold is caused by the fungus Passalora fulva and primarily affects tomato leaves in humid conditions, especially in greenhouse environments.",
        "remedy": "Improve ventilation, reduce humidity, apply fungicides containing copper or chlorothalonil, remove affected leaves.",
        "severity": "Medium",
        "prevention": "Ensure good air circulation, avoid overhead watering, maintain proper humidity levels (below 85%), use resistant varieties.",
        "symptoms": ["Yellow spots on upper leaf surface", "Olive-green to brown fuzzy growth on leaf undersides", "Leaf yellowing and drop"],
        "treatment_timeline": "2-3 weeks with proper management",
        "economic_impact": "Moderate impact on yield and fruit quality",
        "favorable_conditions": "High humidity (>85%) and moderate temperatures (70-80°F)"
    },
    "Tomato___Septoria_leaf_spot": {
        "desc": "Septoria leaf spot is caused by Septoria lycopersici and primarily affects tomato foliage, starting from lower leaves and progressing upward.",
        "remedy": "Remove infected lower leaves, apply fungicides containing chlorothalonil or copper, improve air circulation, mulch around plants.",
        "severity": "Medium",
        "prevention": "Practice crop rotation, avoid overhead watering, maintain proper plant spacing, remove plant debris.",
        "symptoms": ["Small circular spots with dark borders", "Gray centers with tiny black specks", "Yellowing and dropping of leaves"],
        "treatment_timeline": "2-3 weeks with consistent treatment",
        "economic_impact": "Can significantly reduce fruit yield if untreated",
        "favorable_conditions": "Warm, humid weather with temperatures between 60-80°F"
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "desc": "Two-spotted spider mites are tiny pests that feed on plant sap, causing stippling damage and potentially severe defoliation under hot, dry conditions.",
        "remedy": "Increase humidity around plants, use insecticidal soap or neem oil, introduce beneficial mites, regularly spray with water to dislodge mites.",
        "severity": "Medium",
        "prevention": "Maintain adequate soil moisture, avoid over-fertilizing with nitrogen, encourage beneficial insects, regular monitoring.",
        "symptoms": ["Fine stippling on leaves", "Webbing on leaves and stems", "Yellow or bronze discoloration", "Leaf drop in severe cases"],
        "treatment_timeline": "1-2 weeks with consistent treatment",
        "economic_impact": "Can reduce photosynthesis and fruit quality",
        "favorable_conditions": "Hot, dry conditions (above 80°F with low humidity)"
    },
    "Tomato___Target_Spot": {
        "desc": "Target spot is caused by Corynespora cassiicola and creates characteristic target-like lesions on leaves, stems, and fruits.",
        "remedy": "Apply fungicides containing azoxystrobin or chlorothalonil, remove infected plant material, improve air circulation.",
        "severity": "Medium",
        "prevention": "Use resistant varieties, practice crop rotation, ensure proper plant spacing, avoid overhead irrigation.",
        "symptoms": ["Circular lesions with concentric rings", "Brown spots with yellow halos", "Lesions on fruits and stems"],
        "treatment_timeline": "2-3 weeks with fungicide treatment",
        "economic_impact": "Moderate yield and quality losses",
        "favorable_conditions": "Warm, humid conditions (75-85°F with high humidity)"
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "desc": "Tomato Yellow Leaf Curl Virus (TYLCV) is transmitted by whiteflies and causes severe yellowing, curling, and stunting of tomato plants.",
        "remedy": "Control whitefly populations with insecticides or yellow sticky traps, remove infected plants, use virus-resistant varieties.",
        "severity": "High",
        "prevention": "Use whitefly-resistant varieties, control whitefly populations, use reflective mulches, screen greenhouse vents.",
        "symptoms": ["Upward curling of leaves", "Yellowing of leaf margins", "Stunted growth", "Reduced fruit production"],
        "treatment_timeline": "No cure - focus on prevention and whitefly control",
        "economic_impact": "Can cause 50-100% yield loss",
        "favorable_conditions": "Warm temperatures with high whitefly populations"
    },
    "Tomato___Tomato_mosaic_virus": {
        "desc": "Tomato mosaic virus causes mottled light and dark green patterns on leaves and can significantly impact plant growth and fruit quality.",
        "remedy": "Remove infected plants immediately, disinfect tools between plants, control aphid vectors, use virus-free seeds.",
        "severity": "High",
        "prevention": "Use certified virus-free seeds, control aphid populations, disinfect tools, avoid handling wet plants.",
        "symptoms": ["Mottled green patterns on leaves", "Stunted growth", "Distorted leaves", "Reduced fruit quality"],
        "treatment_timeline": "No cure - remove infected plants",
        "economic_impact": "Significant yield and quality losses",
        "favorable_conditions": "Spread by aphids and mechanical transmission"
    },
    "Tomato___healthy": {
        "desc": "Your tomato plant is healthy and thriving! No signs of disease or pest damage detected. The plant shows vibrant green foliage and normal growth patterns.",
        "remedy": "Continue current care practices: maintain consistent watering, provide adequate nutrition, ensure good air circulation, and monitor regularly for any changes.",
        "severity": "None",
        "prevention": "Maintain current care routine, practice preventive measures like crop rotation, proper spacing, and integrated pest management.",
        "symptoms": ["Vibrant green leaves", "No spots or discoloration", "Healthy growth patterns", "Normal leaf structure"],
        "treatment_timeline": "No treatment needed - continue monitoring",
        "economic_impact": "Optimal yield potential",
        "favorable_conditions": "Maintain current growing conditions"
    }
}

# Supported file types
SUPPORTED_IMAGE_TYPES = ["jpg", "jpeg", "png", "bmp", "tiff"]

# CSS color schemes
COLOR_SCHEMES = {
    "default": {
        "primary": "#667eea",
        "secondary": "#764ba2", 
        "accent": "#4299e1",
        "success": "#48bb78",
        "warning": "#ed8936",
        "error": "#f56565"
    },
    "nature": {
        "primary": "#2d5a27",
        "secondary": "#5d8b57",
        "accent": "#7cb342",
        "success": "#4caf50",
        "warning": "#ff9800",
        "error": "#f44336"
    }
}
