from flask import Flask, render_template, request, redirect, url_for
import os
from utils import predict_image, generate_gradcam  # updated to import both

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Get label, confidence, and preprocessed image array
    label, confidence, img_array = predict_image(filepath)

    # Generate Grad-CAM heatmap and save it
    gradcam_path = os.path.join(app.config['UPLOAD_FOLDER'], 'gradcam.jpg')
    generate_gradcam(img_array, output_path=gradcam_path)

    return render_template(
        'result.html',
        label=label,
        confidence=confidence,
        image=file.filename,
        gradcam='gradcam.jpg'
    )

if __name__ == '__main__':
    app.run(debug=True)
