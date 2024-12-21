from transformers import pipeline
from PIL import Image
import requests

checkpoint = "openai/clip-vit-large-patch14"

#Function to load the detector from the checkpoint
def load_detector():
    detector = pipeline(model=checkpoint, task="zero-shot-image-classification")
    return detector

#Function to load the image from the url
def load_image(image_url):
    image = Image.open(requests.get(image_url, stream=True).raw)
    return image

#Function to predict the label to which image belongs to 
def make_prediction(detector,image,candidate_labels):
    prediction =  detector(image,candidate_labels)
    return prediction

#Function ti determine the majority label from the predictons
def majority_label(prediction):
    prob = 0.0
    labels=[]
    scores=[]
    for i in range(len(prediction)):
        labels.append(prediction[i]['label'])
        scores.append(prediction[i]['score'])
        if(prob < prediction[i]['score']):
            selected = prediction[i]['label']
            prob = prediction[i]['score']
    
    return selected,labels,scores
