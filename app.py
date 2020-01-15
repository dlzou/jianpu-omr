#!/usr/bin/env python
import os
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import omr

abs_path = os.path.abspath(__file__)
home_dir = os.path.dirname(abs_path)

EXTENSIONS = ['png', 'jpg', 'jpeg', 'webp']
UPLOAD_PATH = os.path.join(home_dir, 'static/uploaded_img')

app = Flask(__name__)
app.config['UPLOAD_PATH'] = UPLOAD_PATH


@app.route('/', methods=['POST'])
def upload():
    f = request.files.get('img', None)

    if f and valid_file(f.filename):
        filename = secure_filename(f.filename)
        filepath = os.path.join(app.config['UPLOAD_PATH'], filename)
        f.save(filepath)
        
        omr.jianpu_to_midi(filepath)

        return render_template('uploaded.html', success=1, img_path='static/uploaded_img/'+filename)

    return render_template('uploaded.html', success=0)

@app.route('/', methods=['GET'])
def uploaded():
    return render_template('index.html')

def valid_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in EXTENSIONS

if __name__ == '__main__':
    app.run()
