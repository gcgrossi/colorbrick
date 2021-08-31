<img src="static/colorbrick_Cover.png">

# colorbrick
A tiny production-ready webapp built in Python Flask 🌶 Primary color analysis in images with OpenCV and Scikit-image.

Built on top of the color detection technique explained in this article:

[![](https://img.shields.io/badge/color%20detection-C10316?style=for-the-badge&logo=shark)](https://gcgrossi.github.io/color-detection/)

## Front-End

written in

[![](https://img.shields.io/badge/Javascript-black?style=for-the-badge&logo=javascript)]()
[![](https://img.shields.io/badge/HTML-730305?style=for-the-badge&logo=html5)]()
[![](https://img.shields.io/badge/Materialize%20CSS-orange?style=for-the-badge&logo=css3)]()

## Back-End
written in:

[![](https://img.shields.io/badge/Python-B5a300?style=for-the-badge&logo=python)]()
[![](https://img.shields.io/badge/flask-730305?style=for-the-badge&logo=flask)]()

performs SLIC clustering algorithm and Region Adjacency Graph merging using mainly the following libraries:

[![](https://img.shields.io/badge/numpy-D28e08?style=for-the-badge&logo=numpy)]()
[![](https://img.shields.io/badge/opencv-110354?style=for-the-badge&logo=opencv)]()
[![](https://img.shields.io/badge/scikit%20image-414141?style=for-the-badge&logo=scikitlearn)]()

## Deployment
the files ```requirements.txt``` and ```app.yaml``` are configured for correct deployment on Google App Engine.

## Colorbrick in Action

### Home page

<img src="assets/Home.PNG">

Accepts upload of an image from the disk or direct access to the webcam (if on mobile).

<img src="assets/Home_Image.PNG">

### Results page

After processing a Landing page with the analysis report is displayed. The graphs are contructed using

[![](https://img.shields.io/badge/plotly%20js-110354?style=for-the-badge&logo=plotly)]()

<img src="assets/result.PNG">

<img src="assets/result_1.PNG">



