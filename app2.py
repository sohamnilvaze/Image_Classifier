from flask import Flask, request, jsonify
from functions import load_detector, load_image, make_prediction, majority_label

app = Flask(__name__)

# Load the detector once to avoid reloading for each request
detector = load_detector()

@app.route('/')
def index():
    return jsonify({"message": "Welcome to the Image Classification API!"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    # Parse JSON input
    try:
        data = request.json

        # Validate input fields
        image_url = data.get('image_url')
        candidate_labels = data.get('candidate_labels')

        if not image_url or not candidate_labels or len(candidate_labels) != 4:
            return jsonify({"error": "Invalid input. Provide an image URL and exactly 4 candidate labels."}), 400

        # Load and process the image
        image = load_image(image_url)

        # Make predictions
        # predictions = make_prediction(detector, image, candidate_labels)
        predictions=detector(image,candidate_labels=candidate_labels)

        # Find the majority label
        majority = majority_label(predictions)

        # Return predictions and majority label
        return jsonify({
            "predictions": predictions,
            "majority_label": majority
        }), 200

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
