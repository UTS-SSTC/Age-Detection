import os
import sys

# Set environment variables to resolve TensorFlow and OpenMP warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Add project root directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))  # demo directory
project_root = os.path.dirname(current_dir)  # project root directory
sys.path.insert(0, project_root)  # add project root to Python path

from flask import Flask, render_template, request, jsonify
from deepface import DeepFace
import base64, re
from io import BytesIO
from PIL import Image
import torch
import numpy as np
from torchvision import transforms

# Import with the correct class name
import scripts.models as sm
from scripts.backbone import FineTunedBackbone
from scripts.regressor import DeepFeatureLGBM

# Setup Flask app and specify template directory
app = Flask(__name__, template_folder='templates')

# Initialize models
device = sm.get_device()
model_paths = {
    'efficientnet_b0': os.path.join(project_root, 'models', 'efficientnet_b0'),
    'efficientnet_b4': os.path.join(project_root, 'models', 'efficientnet_b4'),
    'resnet_50': os.path.join(project_root, 'models', 'resnet_50'),
    'resnext_101': os.path.join(project_root, 'models', 'resnext_101')
}

# Pre-load models dictionary
loaded_models = {}

# Load all backbone models
try:
    print("Loading backbone models...")
    models = sm.load_all_models()
    print("Backbone models loaded successfully")
except Exception as e:
    print(f"Error loading backbone models: {str(e)}")
    models = {}


# Define image transformations
def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


@app.route('/')
def index():
    return render_template('index.html')


def load_model(model_name):
    """Load model if not already loaded"""
    if model_name not in loaded_models:
        try:
            model_path = model_paths.get(model_name)
            if not model_path:
                raise ValueError(f"Unsupported model: {model_name}")
            if model_name not in models:
                raise ValueError(f"Backbone model {model_name} not found in loaded models")

            # Initialize model with default config
            light_gbm_config = {
                'objective': 'regression',
                'boosting_type': 'gbdt',
                'metric': 'l1',
                'learning_rate': 0.05,
                'n_estimators': 1000,
                'num_leaves': 80,
                'max_depth': 10,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.7,
                'bagging_freq': 5,
                'lambda_l1': 0.5,
                'lambda_l2': 1.0,
                'min_gain_to_split': 0.1,
                'min_data_in_leaf': 30,
                'verbosity': -1,
                'random_state': 42
            }

            # Use the EfficientLightGBM class (correct name)
            fine_tuned_model = FineTunedBackbone(models[model_name])
            model = DeepFeatureLGBM(fine_tuned_model=fine_tuned_model, light_gbm_config=light_gbm_config)

            # Extract directory name and base name from path
            model_dir = os.path.dirname(model_path)
            model_basename = os.path.basename(model_path)

            print(f"Attempting to load model - Directory: {model_dir}, Base name: {model_basename}")
            model.load_model(model_dir, model_basename)
            model.eval()

            loaded_models[model_name] = model
            print(f"Model {model_name} loaded successfully")
        except Exception as e:
            print(f"Error loading model {model_name}: {str(e)}")
            return None

    return loaded_models[model_name]


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image data and model selection from frontend
        data = request.get_json()
        image_data = data['image']
        model_name = data.get('model', 'efficientnet_b0')  # Default to efficientnet_b0 if not specified

        # Decode base64 image
        img_str = re.search(r'base64,(.*)', image_data).group(1)
        image_bytes = base64.b64decode(img_str)

        # Save original image temporarily
        temp_original_path = os.path.join(current_dir, 'temp_original.jpg')
        with open(temp_original_path, 'wb') as f:
            f.write(image_bytes)

        # Use DeepFace to extract face only
        face_location = None
        try:
            # Extract face using DeepFace
            try:
                detected_faces = DeepFace.extract_faces(
                    img_path=temp_original_path,
                    target_size=(224, 224),
                    detector_backend='retinaface',
                    enforce_detection=False
                )
            except TypeError as e:
                # If target_size is not a valid parameter, try without it
                print(f"TypeError with target_size: {str(e)}")
                print("Retrying without target_size parameter...")
                detected_faces = DeepFace.extract_faces(
                    img_path=temp_original_path,
                    detector_backend='retinaface',
                    enforce_detection=False
                )

            if detected_faces and len(detected_faces) > 0:
                # Get the face with highest confidence
                detected_face = max(detected_faces, key=lambda x: x['confidence'])
                face_img = detected_face['face']

                # Store face location for frontend
                if 'facial_area' in detected_face:
                    face_location = detected_face['facial_area']
                    print(f"Face location: {face_location}")

                # Convert the numpy array to PIL Image
                if isinstance(face_img, np.ndarray):
                    if face_img.max() <= 1.0:  # Normalized image
                        face_img = (face_img * 255).astype(np.uint8)
                    face_pil = Image.fromarray(face_img)
                    face_pil = face_pil.resize((224, 224), Image.LANCZOS)
                else:
                    # Fallback to original image if face extraction fails
                    face_pil = Image.open(BytesIO(image_bytes)).convert('RGB')
                    face_pil = face_pil.resize((224, 224), Image.LANCZOS)
            else:
                # No face detected, use original image
                face_pil = Image.open(BytesIO(image_bytes)).convert('RGB')
                face_pil = face_pil.resize((224, 224), Image.LANCZOS)
        except Exception as face_error:
            print(f"Error in face detection: {str(face_error)}")
            # Fallback to original image if face extraction fails
            face_pil = Image.open(BytesIO(image_bytes)).convert('RGB')
            face_pil = face_pil.resize((224, 224), Image.LANCZOS)

        # Clean up the temporary file
        if os.path.exists(temp_original_path):
            os.remove(temp_original_path)

        # Now use our custom model for age prediction
        try:
            # Load the selected model
            model = load_model(model_name)

            if model:
                # Preprocess the image
                transform = get_transforms()
                img_tensor = transform(face_pil).unsqueeze(0).to(device)

                # Get predictions from our model
                with torch.no_grad():
                    # Create a simple DataLoader-format iterator
                    def simple_dataloader(tensor):
                        yield tensor, torch.tensor([0])

                    age_prediction = model.predict(simple_dataloader(img_tensor))
                    age = int(age_prediction[0])

                # Return the result with face location if available
                response = {'age': age, 'model_used': model_name}
                if face_location:
                    response['face_location'] = face_location
                return jsonify(response)
            else:
                # Fallback to using DeepFace
                print("Custom model failed, using DeepFace as fallback")
                temp_face_path = os.path.join(current_dir, 'temp_face.jpg')
                face_pil.save(temp_face_path)

                result = DeepFace.analyze(temp_face_path, actions=['age'], enforce_detection=False)
                age = result[0]['age']

                if os.path.exists(temp_face_path):
                    os.remove(temp_face_path)

                # Return the result with face location if available
                response = {'age': int(age), 'model_used': 'deepface'}
                if face_location:
                    response['face_location'] = face_location
                return jsonify(response)

        except Exception as model_error:
            print(f"Error with custom model: {str(model_error)}")

            # Try DeepFace as fallback
            try:
                temp_face_path = os.path.join(current_dir, 'temp_face.jpg')
                face_pil.save(temp_face_path)
                result = DeepFace.analyze(temp_face_path, actions=['age'], enforce_detection=False)
                age = result[0]['age']

                if os.path.exists(temp_face_path):
                    os.remove(temp_face_path)

                # Return the result with face location if available
                response = {'age': int(age), 'model_used': 'deepface'}
                if face_location:
                    response['face_location'] = face_location
                return jsonify(response)
            except Exception as deepface_error:
                print(f"DeepFace fallback also failed: {str(deepface_error)}")
                return jsonify({'error': 'Both custom model and DeepFace failed'})

    except Exception as e:
        print(f"General error: {str(e)}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print(f"Project root directory: {project_root}")
    print(f"Models directory: {os.path.join(project_root, 'models')}")
    print(f"Available models: {list(model_paths.keys())}")
    app.run(debug=True)