# -*- coding: utf-8 -*-
"""
Created on Wed May 19 08:24:31 2021

@author: Giulio Cornelio Grossi, Ph.D.
@email : giulio.cornelio.grossi@gmail.com
"""

from flask import abort, jsonify, make_response
from werkzeug.utils import secure_filename
import imghdr
from skimage import segmentation
from skimage.future import graph
import numpy as np
import os
import cv2
import json
import plotly
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def decode_image(img_str):
    # CV2
    nparr = np.fromstring(img_str, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    return img_np

def read_image(img_str):
    
    # image comes in byte string
    # format needed for cloud deployment
    # convert bytes->np.array for cv2
    img_bgr = decode_image(img_str)
    
    # return converted rgb->bgr (a cv2 speciality)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def color_masks(label_field, image, bg_label=0, bg_color=(0, 0, 0)):
    """Visualise each segment in `label_field` with its mean color in `image`.
    Parameters
    ----------
    label_field : array of int
        A segmentation of an image.
    image : array, shape ``label_field.shape + (3,)``
        A color image of the same spatial shape as `label_field`.
    bg_label : int, optional
        A value in `label_field` to be treated as background.
    bg_color : 3-tuple of int, optional
        The color for the background label
    Returns
    -------
    out : array, same shape and type as `image`
        The output visualization.
    """
    out = np.zeros(label_field.shape + (3,), dtype=image.dtype)
    labels = np.unique(label_field)
    bg = (labels == bg_label)
    if bg.any():
        labels = labels[labels != bg_label]
        mask = (label_field == bg_label).nonzero()
        out[mask] = bg_color
    
    color_info={"color":[],"npixels":[]}
    
    for label in labels:
        mask = (label_field == label).nonzero()
        color = image[mask].mean(axis=0)
        out[mask] = color
        color_info["color"].append(color)
        color_info["npixels"].append(image[mask].shape[0])
    
    color_info["image"] = out
    return color_info

def get_colormap(color_dict,sort=False):
    # sort the number of pixels for each color
    # if sort bool=True else pass the dictionary
    npixels=sorted(color_dict["npixels"],reverse=True) if sort else color_dict["npixels"]

    # initialize a list for the colormap
    colormap = []

    # calculate total pixels for normalization
    pxtot = sum(npixels)

    for i,npx in enumerate(npixels):
        # get sorted index if bool sort=True
        # else get normal index
        idx = color_dict["npixels"].index(npx) if sort else i
    
        # obtain the corresponding
        # color and frequency
        c = color_dict["color"][idx]
        f = color_dict["npixels"][idx]/pxtot
    
        # push a tuple in the color map with
        # (color, frequency)
        colormap.append((c,f))
        
    return colormap

def make_color_plot(colormap,h=1000,w=1000):
    # create a sliced chart with 
    # all the detected colors
    
    # initialize the x-coordinate
    # and bin-width
    x = bw = 0
    
    # initialize a np array with h,w
    colgrid = np.zeros((h,w,3), dtype='uint8')
    
    # loop on colormap elements
    for cm in colormap:
        # 0th component is the color (r,g,b)
        # 1st component is the frequency
        c,f = cm[0],cm[1]
        # the bin width is proportional
        # to the % frequency of the color
        bw=round(w*f)
        # make a slice h,binwidth
        # with color c
        colgrid[:,x:x+bw] = c
        # increment the x-coordinate
        x+=bw
    
    # plot
    fig = px.imshow(colgrid)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(showlegend=False, 
                      plot_bgcolor='rgb(255,255,255)',
                      margin=dict(l=0, r=0, t=50, b=0))
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def make_bar_chart(colormap,topn=10):
    # create a barchart with the top n colors
    # in order of presence
    
    # create a list of 'rgb(r,g,b)' strings
    # 0th component of colormap is (r,g,b)
    rgb = list(map(lambda c: "rgb({},{},{})".format(int(c[0][0]),int(c[0][1]),int(c[0][2])),colormap))
    
    # initialize figure
    fig = go.Figure()
    
    # loop on the first top n elements 
    # of the colormap
    for i,c in enumerate(colormap[:topn]):
        # add a bar chart with y = frequency
        # x = 'r,g,b' 
        # marker color = (r,g,b)
        fig.add_trace(go.Bar(
            x=[rgb[i]],
            y=[c[1]*100],
            name='',
            marker_color=rgb[i],
            marker_line_color='rgb(0,0,0)'
        ))
        
    # plot figure
    fig.update_layout(showlegend=False, 
                      yaxis=dict(title='[%]'),
                      plot_bgcolor='rgb(255,255,255)',
                      margin=dict(l=0, r=0, t=50, b=0))
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def make_comparison_plot(img1,img2):
    
    # image comparisons
    fig = make_subplots(rows=2, cols=1)
    fig.add_trace(px.imshow(img1).data[0],row=1, col=1)
    fig.add_trace(px.imshow(img2).data[0],row=2, col=1)
    #fig.add_trace(px.imshow(out2).data[0],row=1, col=3)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(showlegend=False, 
                      plot_bgcolor='rgb(255,255,255)',
                      margin=dict(l=0, r=0, t=50, b=0))
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def resize(image):
    
    # define a resizing dictionary based on image area
    scale_dict={"area":[0,6.4E+5,10E+5],"scale":[100,50,30]}
    
    # obtain image area
    image_area = image.shape[0]*image.shape[1]
    
    # init scale %
    scale_percent=scale_dict["scale"][-1]
    
    # select scale % by finding the corresponding
    # range in the scale dictionary
    for i in range(0,len(scale_dict["area"])-1):
        if image_area>=scale_dict["area"][i] and image_area<scale_dict["area"][i+1]: 
            scale_percent = scale_dict["scale"][i]
    
    # calculate scaling dimensions
    # using percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
  
    # resize image
    scaled = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
        
    return scaled

def process(filename):
    
    # read input image
    img_origin = read_image(filename)
    
    # scale input image
    img = resize(img_origin)
    
    print("[INFO] resized image to {}".format(img.shape))
    
    # get the labels corresponding to the clustered pixels
    labels1 = segmentation.slic(img, compactness=30, n_segments=300)

    # process the labels to obtain the
    # processed image and color information
    color_info = color_masks(labels1, img)
    out1 = color_info["image"]

    # N.B orginal Sci-kit image function
    #out1 = color.label2rgb(labels1, img, kind='avg')
    
    # create a RAG
    g = graph.rag_mean_color(img, labels1)

    # merge pixels with mean color distance < threshold
    labels2 = graph.cut_threshold(labels1, g, 20)

    # process the labels to obtain the
    # processed image and color information
    color_info2 = color_masks(labels2, img)
    out2 = color_info2["image"]
    out2 = segmentation.mark_boundaries(out2, labels2, (0, 0, 0))

    # N.B orginal Sci-kit image function
    #out2 = color.label2rgb(labels2, img, kind='avg')
    
    # get colormap 
    colormap = get_colormap(color_info2,sort=True)
    
    # initialize a list of dictionaries with
    # graph info and json content
    outgraph = []
    
    # append the plotply graphs to list
    outgraph.append({"name":"color composition","content": make_color_plot(colormap)})
    outgraph.append({"name":"color prevalence" ,"content": make_bar_chart(colormap)})
    outgraph.append({"name":"image comparison" ,"content": make_comparison_plot(img,out1)})
    
    return outgraph
'''
def _build_cors_prelight_response():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add('Access-Control-Allow-Headers', "*")
    response.headers.add('Access-Control-Allow-Methods', "*")
    return response

def _corsify_actual_response(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response
'''

def validate_image(stream):
    header = stream.read(512)
    stream.seek(0) 
    format = imghdr.what(None, header)
    if not format:
        return None
    return '.' + (format if format != 'jpeg' else 'jpg')

def api_create_order(request):
    
    if request.method == "OPTIONS": # CORS preflight
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': '*',
            'Access-Control-Allow-Headers': '*',
            'Access-Control-Max-Age': '3600'
            }
        return ('', 204, headers)
    
    # Set CORS headers for the main request
    headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': '*',
            'Access-Control-Allow-Headers': '*',
            'Access-Control-Max-Age': '3600'
            }
    
    '''
    # get uploaded file from request
    uploaded_file = request.files['file']
    filename = secure_filename(uploaded_file.filename)
    
    # check if filename exists
    if filename != '':
        # get file extension
        # check if file extension is valid
        file_ext = os.path.splitext(filename)[1]
        if file_ext not in UPLOAD_EXTENSIONS or \
                file_ext != validate_image(uploaded_file.stream):
            
            # abort if not valid (tmp solution)
            abort(400)

    # process the uploaded image and get 
    # plotly json graphs to pass to html
    graphs = process(uploaded_file.read())
    '''
    
    UPLOAD_EXTENSIONS = ['.jpg', '.png', '.gif']
    response = make_response("Mah",200,headers)

    if request.method == "POST":    
        # This code will process each non-file field in the form
        fields = {}
        data = request.form.to_dict()
        for field in data:
            fields[field] = data[field]
            print('Processed field: %s' % field)

        # This code will process each file uploaded
        files = request.files.to_dict()
        for file_name, file in files.items():
            # Note: GCF may not keep files saved locally between invocations.
            # If you want to preserve the uploaded files, you should save them
            # to another location (such as a Cloud Storage bucket).
            #graphs = process(file.read())
            print('Processed file: %s' % file_name)

        response = make_response(jsonify(files),200,headers)

    return response

