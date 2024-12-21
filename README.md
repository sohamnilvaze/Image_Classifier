
# Image Classification Web Application

This project implements a Flask-based web application for image classification. Users can upload an image URL, provide candidate labels, and view the predicted label along with a comparison of the confidence scores for the candidate labels. The application uses OpenAI's CLIP model for zero-shot image classification.
## Features

- **Image Upload:** Accepts an image URL provided by the user.
- **Candidate Labels:** Users input up to four labels to classify the uploaded image.
- **Prediction:** Uses the CLIP model to predict the best match for the image among the provided labels.
- **REST API**



## Deployment

To run the Flask app locally on your machine

1)Clone the repository
```bash
  git clone https://github.com/sohamnilvaze/Image_Classifier.git
```
2)Install the required dependencies:
```bash
  pip install -r requirements.txt
```
3)Start the Flask app:
```bash
  python3 app.py
```
4)Open your browser and navigate to http://127.0.0.1:5000.

## Tech Stack

**Frontend:** HTML, CSS, Javascript

**Backend:** Flask(Python)

**Model:** OpenAI’s CLIP model for zero-shot image classification

**Libraries:** Tensorflow, Keras, Transformers


## Demo

A video which gives a fair idea about the working of the project

https://drive.google.com/file/d/1iDLtMuaexVNEi6MMWq8kNI2Cj5QuLtF9/view?usp=sharing
