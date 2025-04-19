import os
import base64
import re
from io import BytesIO
from PIL import Image

from flask import Flask, render_template, request, jsonify
import torch
from torchvision import transforms
import torch.nn as nn

# Setup Flask with template path
template_path = os.path.join(os.path.dirname(__file__), 'templates')
app = Flask(__name__, template_folder=template_path)

# Choose device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== Define mock model =====
# Load pretrained EfficientNet-B0 from TorchHub
feature_model = torch.hub.load(
    'NVIDIA/DeepLearningExamples:torchhub',
    'nvidia_efficientnet_b0',
    pretrained=True,
    trust_repo=True
)

# Freeze original model
for param in feature_model.parameters():
    param.requires_grad = False

# Add fake age regression head
class AgeMockModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.flatten = nn.Flatten()
        self.fake_head = nn.Linear(1000, 1)  # Simulated regression head

    def forward(self, x):
        x = self.backbone(x)
        x = self.flatten(x)
        x = self.fake_head(x)
        return x

model = AgeMockModel(feature_model).to(device)
model.eval()

# Preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        image_data = data['image']
        img_str = re.search(r'base64,(.*)', image_data).group(1)
        image_bytes = base64.b64decode(img_str)
        image = Image.open(BytesIO(image_bytes)).convert('RGB')

        # Apply transform
        input_tensor = transform(image).unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            output = model(input_tensor)
            predicted_age = output.item()

        return jsonify({'age': round(predicted_age, 2)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the server
if __name__ == '__main__':
    app.run(debug=True)