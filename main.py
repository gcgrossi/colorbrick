# -*- coding: utf-8 -*-
"""
Created on Tue May 18 09:44:19 2021

@author: Giulio Cornelio Grossi, Ph.D.
@email : giulio.cornelio.grossi@gmail.com
"""

from flask import Flask
from flask_material import Material
from flask import render_template, request, redirect, url_for, abort, flash
from werkzeug.utils import secure_filename
import colors
import imghdr
import os
import googlecloudprofiler

app = Flask(__name__)
Material(app)

app.config['MAX_CONTENT_LENGTH'] = 2400 * 2400
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.jpeg','.png', '.gif']
app.config['UPLOAD_PATH'] = './'

def validate_image(stream):
    header = stream.read(512)
    stream.seek(0) 
    format = imghdr.what(None, header)
    if not format:
        return None
    return '.' + (format if format != 'jpeg' else 'jpg')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_file():
    
    # start_profiler
    try:
        googlecloudprofiler.start(
            service='hello-profiler',
            # verbose is the logging level. 0-error, 1-warning, 2-info,
            # 3-debug. It defaults to 0 (error) if not set.
            verbose=3,
            project_id='color-detection-app'
            # project_id must be set if not running on GCP.
            # project_id='my-project-id',
        )
    except (ValueError, NotImplementedError) as exc:
        print(exc)  # Handle errors here   
    
    # get uploaded file from request
    uploaded_file = request.files['file']
    filename = secure_filename(uploaded_file.filename)
    
    # check if filename exists
    if filename != '':
        # get file extension
        # check if file extension is valid
        file_ext = os.path.splitext(filename)[1]
        print("[INFO] file extension:"+file_ext)
        if file_ext not in app.config['UPLOAD_EXTENSIONS']:
        #or file_ext != validate_image(uploaded_file.stream):
            print("[CRITICAL] "+file_ext +" is a bad file format!")
        else: 
            print("[INFO] "+file_ext +" is a good file format")
            # abort if not valid (tmp solution)
            #abort(400)

    # process the uploaded image and get 
    # plotly json graphs to pass to html
    graphs = colors.process(uploaded_file.read())


    # render html with results
    return render_template('report.html', graphJSON=graphs) #redirect(url_for('index'))


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)

    
    