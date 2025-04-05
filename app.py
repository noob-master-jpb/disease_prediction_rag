from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
from vertexai.generative_models import GenerativeModel
from google.cloud import bigquery
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from functools import wraps

app = Flask(__name__)
CORS(app) 

load_dotenv()
project_id = os.getenv("project_id")
dataset_id = os.getenv("dataset_id")
table_id = os.getenv("table_id")
model = SentenceTransformer(os.getenv("model"))
gemini_model = GenerativeModel(os.getenv("gemini_model"))
port = os.getenv("port")
client = bigquery.Client(project=project_id)

# Load the Bearer Token from environment variables
bearer_token = os.getenv("BEARER_TOKEN")

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return jsonify({"error": "Authorization header missing or invalid"}), 401
        
        token = auth_header.split(" ")[1]
        if token != bearer_token:
            return jsonify({"error": "Invalid or missing token"}), 403
        
        return f(*args, **kwargs)
    return decorated

def generate_query_embedding(user_input):
    inputs = f"[CLS] {user_input}"
    
    embedding = model.encode(inputs)
    
    return embedding.tolist()
def retrieve_similar_cases(query_embedding, top_k=1000):
    embedding_sql = "[" + ", ".join(f"{x:.6f}" for x in query_embedding) + "]"
    
    query = f"""
        SELECT disease, text 
        FROM `{project_id}.{dataset_id}.disease_embeddings`
        ORDER BY ML.DISTANCE(embedding, {embedding_sql}, 'COSINE')
        LIMIT {top_k}
    """
    
    results = client.query(query).result()
    return [dict(row) for row in results]
def predict_disease(user_input):
    query_embedding = generate_query_embedding(user_input)

    cases = retrieve_similar_cases(query_embedding)

    context = "\n".join([f"- Disease: {case['disease']}, Description: {case['text']}" for case in cases])

    prompt = f"""**Your Role:** You are 'QuickAID', an AI health assistant. Your purpose is to understand user-reported symptoms and, using the provided context cases, suggest *potential* related conditions. You must be empathetic and helpful. You will respond with *only* a valid JSON object containing 'response', 'thinking', and 'reasoning' fields.

**Critical Limitations:**
* You are NOT a substitute for a professional medical diagnosis or advice.
* You cannot provide medical treatment recommendations.
* Your knowledge is based on the provided context and general patterns.

**Context Cases:** Below are descriptions/symptoms associated with certain diseases. These *might* be relevant to the user's situation. Use them cautiously as reference points.
--- START CONTEXT ---
{context}
--- END CONTEXT ---

**User's Input:**
"{user_input}"

**Your Task and Response Logic:**

1. **Analyze Input:** First, determine the nature of the user's input.
    * Does it clearly describe specific physical or mental health symptoms?
    * Is it vague or general?
    * Do use your own knowledge besides the context.
    * Do not hallucinate.
    * Do not produce bad or unrelated output.

2. **Response Generation (as a valid JSON object with 'response_type', 'disease', and 'definition'):**

    * **If Input Contains Specific Symptoms:**
        Output *only* the following valid JSON object:
        ```json
        {{
            "response_type": "prediction",
            "disease": {{
                "1": "...",
                "2": "...",
                "3": "...",
                "4": "...",
                "5": "..."
            }},
            "definition": {{
                "1": "...",
                "2": "...",
                "3": "...",
                "4": "...",
                "5": "..."
            }}
        }}
        ```
        For the `"disease"`:
        a. Identify the top 5 most plausible conditions suggested by the combination of symptoms and context, ranked as 1, 2, 3, 4, and 5.
        For the `"definition"`:
        b. Provide a brief definition of each of the top 5 identified diseases, corresponding to their rank.
        c. The definition should be to the point and informative, without unnecessary details and try to keep it short.


    * **If Input is Vague/General or Conversational/Unrelated:**
        Output *only* the following valid JSON object:
        ```json
        {{
            "response_type": "error",
            "message": "no symptoms found.. give a better prompt"
        }}
        ```

**Output Style:** Ensure the output is a *valid JSON object* with the specified fields ('response', 'thinking', 'reasoning') and without any additional text or markdown.
"""



    try:
        response_text = gemini_model.generate_content(prompt).text

        if response_text.startswith("```json") and response_text.endswith("```"):
            response_text = response_text[7:-3].strip()  

        try:
            response_json = json.loads(response_text)
            return response_json
        except json.JSONDecodeError as e:
            print(f"JSON Decode Error: {e}")
            print(f"Raw Response: {response_text}")
            return {"response_type": "error", "message": "Error decoding JSON response from the model.", "raw_response": response_text}
    except Exception as e:
        return {"response_type": "error", "message": f"An error occurred while calling the Gemini API: {e}"}



@app.route("/ai/text", methods=["POST"])
@token_required
def ai_text():
    try:
        text = request.json["prompt"]
        if not text:
            return jsonify({"error": "No prompt provided"}), 400
        response = predict_disease(text)
        return jsonify(response), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(port), debug=True)