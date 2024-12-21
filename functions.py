from transformers import pipeline
from PIL import Image
import requests

checkpoint = "openai/clip-vit-large-patch14"

def load_detector():
    detector = pipeline(model=checkpoint, task="zero-shot-image-classification")
    return detector

def load_image(image_url):
    image = Image.open(requests.get(image_url, stream=True).raw)
    return image

def make_prediction(detector,image,candidate_labels):
    prediction =  detector(image,candidate_labels)
    return prediction

def majority_label(prediction):
    prob = 0.0
    for i in range(len(prediction)):
        if(prob < prediction[i]['score']):
            selected = prediction[i]['label']
            prob = prediction[i]['score']
    
    return selected
