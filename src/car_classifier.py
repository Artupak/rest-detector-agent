import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import cv2
import os
import json
from functools import lru_cache
from pathlib import Path

class CarClassifier:
    """Class for classifying car makes and models using ResNet50."""
    
    def __init__(self, fine_tune=False):
        """Initialize the car classification model.
        
        Args:
            fine_tune: Whether to use fine-tuned model (if available)
        """
        # Load car database
        self.car_db = self._load_car_database()
        
        # Create reverse mapping for model lookup
        self.model_to_make = {}
        for category in self.car_db:
            for make, models in self.car_db[category].items():
                for model in models:
                    self.model_to_make[model.lower()] = (make, model)
        
        # Input shape required by the model
        self.input_shape = (224, 224)
        
        # Initialize model
        if fine_tune and self._fine_tuned_model_exists():
            self.model = self._load_fine_tuned_model()
        else:
            self.model = ResNet50(weights='imagenet')
            
        # Initialize cache
        self.prediction_cache = {}
    
    def _load_car_database(self):
        """Load car make/model database."""
        db_path = Path(__file__).parent / 'data' / 'car_models.json'
        try:
            with open(db_path) as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading car database: {str(e)}")
            return {}
    
    def _fine_tuned_model_exists(self):
        """Check if fine-tuned model exists."""
        model_path = Path(__file__).parent / 'models' / 'car_classifier_fine_tuned'
        return model_path.exists()
    
    def _load_fine_tuned_model(self):
        """Load fine-tuned model."""
        model_path = Path(__file__).parent / 'models' / 'car_classifier_fine_tuned'
        return tf.keras.models.load_model(model_path)
    
    def fine_tune(self, dataset_path, epochs=10):
        """Fine-tune the model on Stanford Cars Dataset.
        
        Args:
            dataset_path: Path to Stanford Cars Dataset
            epochs: Number of training epochs
        """
        # Load dataset
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            dataset_path / 'train',
            image_size=self.input_shape,
            batch_size=32
        )
        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            dataset_path / 'val',
            image_size=self.input_shape,
            batch_size=32
        )
        
        # Create fine-tuning model
        base_model = ResNet50(weights='imagenet', include_top=False)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(len(train_ds.class_names), activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        
        # Freeze base layers
        for layer in base_model.layers:
            layer.trainable = False
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train model
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs
        )
        
        # Save model
        model_path = Path(__file__).parent / 'models' / 'car_classifier_fine_tuned'
        model.save(model_path)
        self.model = model
    
    @lru_cache(maxsize=100)
    def preprocess_image(self, img_bytes: bytes) -> np.ndarray:
        """Preprocess image for the model.
        
        Args:
            img_bytes: Image as bytes
            
        Returns:
            Preprocessed image
        """
        # Convert bytes to numpy array
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize
        img_resized = cv2.resize(img_rgb, self.input_shape)
        
        # Convert to array and add batch dimension
        img_array = img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Preprocess for ResNet50
        return preprocess_input(img_array)
    
    def _match_car_model(self, class_name: str) -> tuple:
        """Match predicted class to car make/model.
        
        Args:
            class_name: Predicted class name
            
        Returns:
            Tuple of (make, model) or (None, None)
        """
        # Try exact match first
        class_lower = class_name.lower()
        for model_name, (make, model) in self.model_to_make.items():
            if model_name in class_lower:
                return make, model
        
        # Try partial matches
        for model_name, (make, model) in self.model_to_make.items():
            model_parts = model_name.split()
            if any(part.lower() in class_lower for part in model_parts):
                return make, model
        
        return None, None
    
    def predict(self, img: np.ndarray) -> dict:
        """Predict car make and model.
        
        Args:
            img: BGR image as numpy array
            
        Returns:
            Dictionary containing:
                - make: car manufacturer
                - model: car model
                - confidence: confidence score
                - category: car category (sedan, suv, etc.)
        """
        try:
            # Convert image to bytes for caching
            _, img_bytes = cv2.imencode('.jpg', img)
            img_bytes = img_bytes.tobytes()
            
            # Check cache
            if img_bytes in self.prediction_cache:
                return self.prediction_cache[img_bytes]
            
            # Preprocess image
            processed_img = self.preprocess_image(img_bytes)
            
            # Get predictions
            predictions = self.model.predict(processed_img, verbose=0)
            
            # Get top prediction
            if isinstance(self.model, ResNet50):
                # Using base ResNet50
                pred_class = tf.keras.applications.resnet50.decode_predictions(predictions, top=1)[0][0]
                class_name, confidence = pred_class[1], pred_class[2]
                make, model = self._match_car_model(class_name)
            else:
                # Using fine-tuned model
                class_idx = np.argmax(predictions[0])
                confidence = float(predictions[0][class_idx])
                class_name = self.model.class_names[class_idx]
                make, model = self._match_car_model(class_name)
            
            # Find category
            category = None
            if make and model:
                for cat, makes in self.car_db.items():
                    if make in makes and model in makes[make]:
                        category = cat
                        break
            
            result = {
                'make': make if make else 'Unknown',
                'model': model if model else class_name,
                'confidence': float(confidence),
                'category': category if category else 'Unknown'
            }
            
            # Cache result
            self.prediction_cache[img_bytes] = result
            
            return result
            
        except Exception as e:
            print(f"Error during car classification: {str(e)}")
            return {
                'make': None,
                'model': None,
                'confidence': 0.0,
                'category': None
            } 