from flask import Flask, render_template, request
from functions import load_detector, load_image,make_prediction, majority_label
import matplotlib.pyplot as plt

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
    predictions,labels,scores=detector(image,candidate_labels=candidate_labels)

    plt.figure(figsize=(8, 6))
    plt.bar(labels, scores, color='skyblue', edgecolor='black')

    # Add labels and title
    plt.xlabel('Labels', fontsize=14)
    plt.ylabel('Scores', fontsize=14)
    plt.title('Scores for Each Label', fontsize=16)
    plt.ylim(0, 1)  # Assuming scores are between 0 and 1
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('/plots/bar_chart.png',dpi=300, bbox_inches='tight')

    majority=majority_label(predictions)
    return render_template('result.html',
                           image_url = image_url,  
                           predicted_label=predictions,  
                           match=majority,
                           labels=labels,
                           scores=scores)


if __name__ == '__main__':
    app.run(debug=True)