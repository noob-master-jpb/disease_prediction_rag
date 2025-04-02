from flask import Flask, request, jsonify,session
from rag import *

app = Flask(__name__)

@app.route("/ai/text", methods=["POST"])
def ai_text():
    try:
        text = request.json["prompt"]
        if not text:
            return jsonify({"error": "No prompt provided"}), 400
        response = predict_disease(text)
        return jsonify(response), 200
    except:
        return 400