from flask import Flask, render_template, request, jsonify
from deepface import DeepFace
import os, base64, re
from io import BytesIO

# Setup Flask app and specify template directory
app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), 'templates'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image data from the frontend
        data = request.get_json()
        image_data = data['image']
        img_str = re.search(r'base64,(.*)', image_data).group(1)
        image_bytes = base64.b64decode(img_str)

        # Save the image temporarily
        temp_path = 'temp.jpg'
        with open(temp_path, 'wb') as f:
            f.write(image_bytes)

        # Analyze the image using DeepFace
        result = DeepFace.analyze(temp_path, actions=['age', 'gender'], enforce_detection=False)
        age = result[0]['age']
        gender = result[0]['dominant_gender']
        os.remove(temp_path)

        return jsonify({'age': int(age), 'gender': gender})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
