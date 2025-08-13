import os
import io
import json
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from google.cloud import vision
import openai
from config import OPENAI_API_KEY, OPENAI_API_KEY_2, GOOGLE_APPLICATION_CREDENTIALS

# Setup logging
logging.basicConfig(level=logging.INFO)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Set credentials for external APIs
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_APPLICATION_CREDENTIALS
openai.api_key = OPENAI_API_KEY_2

# Initialize Google Vision API client
vision_client = vision.ImageAnnotatorClient()

def get_google_vision_labels(image_content):
    """Extract labels from image using Google Cloud Vision."""
    image = vision.Image(content=image_content)
    response = vision_client.label_detection(image=image)

    if response.error.message:
        raise Exception(f"Google Vision API error: {response.error.message}")

    return [label.description for label in response.label_annotations]

def get_openai_classification_and_tip(labels):
    """Classify garbage and get eco tip using OpenAI."""
    prompt = (
        f"You are an expert environmental assistant. Given these labels from an image: {labels}, "
        "classify the garbage item into one of these categories: recyclable, organic, general waste, or hazardous. "
        "Then provide one concise environmental tip related to disposing or reusing this item properly. "
        "Respond with ONLY a single-line JSON object, no list, no explanation, no markdown. Example format: "
        "{\"category\": \"recyclable\", \"tip\": \"Rinse plastic containers before recycling.\"}"
        "Do not add any extra explanation."
    )

    response = openai.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You classify garbage items and give eco-friendly disposal tips."},
        {"role": "user", "content": prompt}
    ],
    temperature=0,
    max_tokens=150
    )


    return response.choices[0].message.content

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "SmartBin AI backend running"}), 200

@app.route("/classify_garbage", methods=["POST"])
def classify_garbage():
    try:
        image_file = request.files.get("image")

        if not image_file or not image_file.mimetype.startswith("image/"):
            logging.warning("Invalid or missing image.")
            return jsonify({
                "category": "general waste",
                "tip": "Dispose of waste responsibly."
            }), 400

        image_content = image_file.read()

        # Process image with Google Vision
        labels = get_google_vision_labels(image_content)
        logging.info(f"Google Vision labels: {labels}")

        # Classify with OpenAI
        classification_json = get_openai_classification_and_tip(labels)
        logging.info(f"OpenAI raw JSON: {classification_json}")

        try:
            classification = json.loads(classification_json)
        except json.JSONDecodeError:
            logging.warning("Failed to decode JSON from OpenAI")
            classification = {
                "category": "general waste",
                "tip": "Dispose of waste responsibly."
            }

        return jsonify({
            "category": classification.get("category", "general waste"),
            "tip": classification.get("tip", "Dispose of waste responsibly.")
        })

    except Exception as e:
        logging.error(f"Unexpected error: {e}", exc_info=True)
        return jsonify({
            "category": "general waste",
            "tip": "Dispose of waste responsibly."
        }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
