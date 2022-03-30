
from flask import Flask, render_template, request, redirect, flash, url_for
from werkzeug.utils import secure_filename
import os
from PIL import Image
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from net import getModel

UPLOAD_FOLDER = r'C:\Users\avkos\Desktop\idp_flask\uploads'
mapper = {
        '0': 'Shih-Tzu',
        '1': 'Rhodesian ridgeback',
        '2': 'Beagle',
        '3': 'English foxhound',
        '4': 'Australian terrier',
        '5': 'Border terrier',
        '6': 'Golden retriever',
        '7': 'Old English sheepdog',
        '8': 'Samoyed',
        '9': 'Dingo'
    }

transform = A.Compose(
    [
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'secret_key'

def getPrediction(filename):
    image = Image.open(filename)
    x = np.array(image.convert('RGB'))
    x_processed = transform(image=x)['image']
    y_hat = model(x_processed.unsqueeze(0))
    preds = y_hat.argmax(dim=1).item()
    return mapper[str(preds)], torch.max(torch.softmax(y_hat, dim=1)).item()*100

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/display/<filename>')
def display_image(filename):
	#print('display_image filename: ' + filename)
	return redirect(url_for('static', filename='uploads/' + filename), code=301)


@app.route('/', methods=['POST'])
def submit_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
            label, acc = getPrediction(os.path.join(app.config['UPLOAD_FOLDER'],filename))
            flash(label)
            flash(acc)
            flash(filename)
            return redirect('/')

if __name__ == '__main__':
    model = getModel()
    app.run()

