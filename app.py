from flask import Flask, render_template, request
from functions import load_detector, load_image,make_prediction, majority_label

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    image_url = request.form['image_url']
    label1=request.form['label1']
    label2=request.form['label2']
    label3=request.form['label3']
    label4=request.form['label4']
    candidate_labels =[]
    candidate_labels.append(label1)
    candidate_labels.append(label2)
    candidate_labels.append(label3)
    candidate_labels.append(label4)

    detector = load_detector()
    image=load_image(image_url)
    
    # predictions= make_prediction(detector,image, candidate_labels)
    predictions=detector(image,candidate_labels=candidate_labels)
    majority=majority_label(predictions)
    return render_template('result.html',
                           image_url = image_url,  
                           predicted_label=predictions,  
                           match=majority)

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'image_file' not in request.files:
#         return "No image uploaded", 400

#     image_file = request.files['image_file'].read()
#     image = Image.open(file.stream).convert("RGB")
#     label1=request.form['label1']
#     label2=request.form['label2']
#     label3=request.form['label3']
#     label4=request.form['label4']
#     candidate_labels =[]
#     candidate_labels.append(label1)
#     candidate_labels.append(label2)
#     candidate_labels.append(label3)
#     candidate_labels.append(label4)

#     detector = load_detector()
#     # image=load_image(image_url)
    
#     # predictions= make_prediction(detector,image_file, candidate_labels)
#     predictions=detector(image_file,candidate_labels=candidate_labels)
#     majority = majority_label(predictions)
#     return render_template('result.html',   
#                            match=majority)

if __name__ == '__main__':
    app.run(debug=True)