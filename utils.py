"""
Utility functions for AgroVision Pro backend operations
"""

import os
import json
import pickle
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
import tensorflow as tf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ImageProcessor:
    """Enhanced image processing utilities"""
    
    @staticmethod
    def enhance_image(image: Image.Image) -> Image.Image:
        """Apply image enhancements for better model prediction"""
        try:
            # Convert to RGB if not already
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Enhance contrast slightly
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.1)
            
            # Enhance sharpness slightly
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.1)
            
            # Apply slight noise reduction
            image = image.filter(ImageFilter.MedianFilter(size=3))
            
            return image
            
        except Exception as e:
            logger.error(f"Error enhancing image: {str(e)}")
            return image
    
    @staticmethod
    def validate_image(image: Image.Image) -> Tuple[bool, str]:
        """Validate if image is suitable for analysis"""
        try:
            # Check image size
            if image.size[0] < 50 or image.size[1] < 50:
                return False, "Image too small (minimum 50x50 pixels)"
            
            # Check if image is too large
            if image.size[0] > 4000 or image.size[1] > 4000:
                return False, "Image too large (maximum 4000x4000 pixels)"
            
            # Check if image has sufficient detail (not completely uniform)
            np_img = np.array(image)
            if np.std(np_img) < 10:
                return False, "Image appears to lack sufficient detail"
            
            return True, "Image is valid"
            
        except Exception as e:
            logger.error(f"Error validating image: {str(e)}")
            return False, f"Validation error: {str(e)}"
    
    @staticmethod
    def preprocess_for_model(image: Image.Image, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """Preprocess image for model prediction"""
        try:
            # Resize image
            image = image.resize(target_size, Image.Resampling.LANCZOS)
            
            # Convert to numpy array
            img_array = np.array(image, dtype=np.float32)
            
            # Normalize pixel values
            img_array = img_array / 255.0
            
            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            
            return img_array
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise

class ModelManager:
    """Enhanced model management utilities"""
    
    @staticmethod
    def validate_model_files(model_path: str, class_names_path: str) -> Tuple[bool, str]:
        """Validate that model files exist and are readable"""
        try:
            # Check model file
            if not os.path.exists(model_path):
                return False, f"Model file not found: {model_path}"
            
            if not model_path.endswith('.h5'):
                return False, "Model file must be in .h5 format"
            
            # Check class names file
            if not os.path.exists(class_names_path):
                return False, f"Class names file not found: {class_names_path}"
            
            if not class_names_path.endswith('.pkl'):
                return False, "Class names file must be in .pkl format"
            
            # Try to load class names
            try:
                with open(class_names_path, 'rb') as f:
                    class_names = pickle.load(f)
                if not isinstance(class_names, (list, tuple)):
                    return False, "Class names file must contain a list or tuple"
            except Exception as e:
                return False, f"Error loading class names: {str(e)}"
            
            return True, "Model files are valid"
            
        except Exception as e:
            logger.error(f"Error validating model files: {str(e)}")
            return False, f"Validation error: {str(e)}"
    
    @staticmethod
    def get_model_info(model_path: str) -> Dict:
        """Get basic information about the model file"""
        try:
            file_size = os.path.getsize(model_path)
            file_size_mb = file_size / (1024 * 1024)
            
            modified_time = datetime.fromtimestamp(os.path.getmtime(model_path))
            
            return {
                'file_size_mb': round(file_size_mb, 2),
                'last_modified': modified_time.isoformat(),
                'file_path': model_path
            }
            
        except Exception as e:
            logger.error(f"Error getting model info: {str(e)}")
            return {}

class DataManager:
    """Data persistence and management utilities"""
    
    @staticmethod
    def save_analytics_data(data: Dict, filepath: str = "analytics_data.json") -> bool:
        """Save analytics data to file"""
        try:
            # Convert numpy arrays to lists for JSON serialization
            json_data = {}
            for key, value in data.items():
                if isinstance(value, np.ndarray):
                    json_data[key] = value.tolist()
                elif isinstance(value, (np.int64, np.int32)):
                    json_data[key] = int(value)
                elif isinstance(value, (np.float64, np.float32)):
                    json_data[key] = float(value)
                else:
                    json_data[key] = value
            
            with open(filepath, 'w') as f:
                json.dump(json_data, f, indent=2)
            
            logger.info(f"Analytics data saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving analytics data: {str(e)}")
            return False
    
    @staticmethod
    def load_analytics_data(filepath: str = "analytics_data.json") -> Dict:
        """Load analytics data from file"""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    data = json.load(f)
                logger.info(f"Analytics data loaded from {filepath}")
                return data
            else:
                logger.info("No existing analytics data found")
                return {}
                
        except Exception as e:
            logger.error(f"Error loading analytics data: {str(e)}")
            return {}
    
    @staticmethod
    def export_history_to_csv(history_data: List[Dict], filepath: str) -> bool:
        """Export diagnosis history to CSV file"""
        try:
            df = pd.DataFrame(history_data)
            df.to_csv(filepath, index=False)
            logger.info(f"History exported to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting history: {str(e)}")
            return False

class PerformanceMonitor:
    """Monitor application performance and usage"""
    
    def __init__(self):
        self.start_time = None
        self.metrics = {}
    
    def start_timer(self, operation: str):
        """Start timing an operation"""
        self.start_time = datetime.now()
        self.metrics[operation] = {'start_time': self.start_time}
    
    def end_timer(self, operation: str) -> float:
        """End timing and return duration in seconds"""
        if operation in self.metrics and 'start_time' in self.metrics[operation]:
            end_time = datetime.now()
            duration = (end_time - self.metrics[operation]['start_time']).total_seconds()
            self.metrics[operation]['duration'] = duration
            self.metrics[operation]['end_time'] = end_time
            return duration
        return 0.0
    
    def get_metrics(self) -> Dict:
        """Get all collected metrics"""
        return self.metrics

class CacheManager:
    """Simple caching for model predictions and analytics"""
    
    def __init__(self, max_size: int = 100):
        self.cache = {}
        self.max_size = max_size
        self.access_times = {}
    
    def get(self, key: str) -> Optional[any]:
        """Get item from cache"""
        if key in self.cache:
            self.access_times[key] = datetime.now()
            return self.cache[key]
        return None
    
    def set(self, key: str, value: any):
        """Set item in cache"""
        # Remove oldest items if cache is full
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
        
        self.cache[key] = value
        self.access_times[key] = datetime.now()
    
    def clear(self):
        """Clear all cache"""
        self.cache.clear()
        self.access_times.clear()

# Global instances
image_processor = ImageProcessor()
model_manager = ModelManager()
data_manager = DataManager()
performance_monitor = PerformanceMonitor()
cache_manager = CacheManager()

# Utility functions
def get_file_hash(filepath: str) -> str:
    """Get a simple hash of file for caching purposes"""
    try:
        import hashlib
        with open(filepath, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        return file_hash
    except:
        return str(datetime.now().timestamp())

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0B"
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    return f"{size_bytes:.1f}{size_names[i]}"

def validate_confidence_threshold(threshold: float) -> bool:
    """Validate confidence threshold value"""
    return 0.0 <= threshold <= 1.0

def get_system_info() -> Dict:
    """Get basic system information"""
    try:
        import platform
        import psutil
        
        return {
            'platform': platform.system(),
            'platform_version': platform.version(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'memory_gb': round(psutil.virtual_memory().total / (1024**3), 2),
            'tensorflow_version': tf.__version__
        }
    except ImportError:
        return {
            'platform': 'Unknown',
            'tensorflow_version': tf.__version__
        }
