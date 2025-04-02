from flask import Flask, request, jsonify,session
import os
import json
from vertexai.generative_models import GenerativeModel
from google.cloud import bigquery
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()
project_id = os.getenv("project_id")
dataset_id = os.getenv("dataset_id")
table_id = os.getenv("table_id")
model = SentenceTransformer(os.getenv("model"))
gemini_model = GenerativeModel(os.getenv("gemini_model"))

client = bigquery.Client(project=project_id)

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

def generate_query_embedding(user_input):
    inputs = f"[CLS] {user_input}"
    
    embedding = model.encode(inputs)
    
    return embedding.tolist()
def predict_disease(user_input, history_text):
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
**Conversation History:**
{history_text}

**User's Input:**
"{user_input}"

**Your Task and Response Logic:**

1. **Analyze Input:** First, determine the nature of the user's input.
    * Does it clearly describe specific physical or mental health symptoms?
    * Is it vague or general?
    * Is it conversational or unrelated to health?
    * Do use your own knowledge besides the context.
    * Ask as many questions you need to ask to the user.
    * Identify yourself as QuickAID.
    * Do not hallucinate.
    * Do not produce bad or unrelated output.
    * Try to be as professonal as possible
    * Do describe about the prediction the user asks about it.
    * If the case is too severe.
    * If somethings are unclear to you, ask the user to describe it and help the user describe it by guiding them.

2. **Response Generation (as a valid JSON object with 'response', 'thinking', and 'reasoning'):**

    * **If Input Contains Specific Symptoms:**
        Output *only* the following valid JSON object:
        ```json
        {{
          "response": "...",
          "thinking": "...",
          "reasoning": "..."
        }}
        ```
        For the `"response"`:
        a. Acknowledge the user's symptoms empathetically.
        b. Identify the *single most plausible* condition suggested by the combination of symptoms and context. If multiple seem equally likely based on context, mention the top one or two possibilities briefly.
        c. Ask 1-2 specific, relevant follow-up questions to gather more details.
        For the `"thinking"`:
        d. Briefly explain that you considered the symptoms alongside the provided Context Cases.
        For the `"reasoning"`:
        e. Explain your reasoning concisely, referencing how the user's symptoms relate to the context if there's a clear link.

    * **If Input is Vague/General (e.g., "I feel unwell"):**
        Output *only* the following valid JSON object:
        ```json
        {{
          "response": "...",
          "thinking": "...",
          "reasoning": "..."
        }}
        ```
        For the `"response"`:
        a. Respond empathetically, acknowledging they don't feel well.
        b. Gently guide them to provide more specific information by asking open-ended questions. For example: "I understand you're feeling unwell. Could you tell me more about what specific symptoms you are experiencing? For instance, are you feeling any pain, fatigue, fever, or anything else in particular?"
        For the `"thinking"`:
        Indicate that the input was vague or general and more specific information is needed to provide relevant context.
        For the `"reasoning"`:
        State that a potential condition cannot be identified based on the vague input alone.

    * **If Input is Conversational/Unrelated (e.g., "hello"):**
        Output *only* the following valid JSON object:
        ```json
        {{
          "response": "...",
          "thinking": "...",
          "reasoning": "..."
        }}
        ```
        For the `"response"`:
        a. Respond in a friendly and human-like manner. For example: "Hello! I hope you're doing well. If you have any health concerns or are experiencing any symptoms you'd like to discuss, please feel free to share them with me. I'm here to help provide some information based on the data I have."
        b. Briefly explain your purpose and encourage them to share health-related information.
        For the `"thinking"`:
        Indicate that the input was conversational or unrelated to health and a health-related query is expected.
        For the `"reasoning"`:
        State that the input was not a description of health symptoms.

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





app = Flask(__name__)
app.secret_key = 'your_secret_key'

@app.route('/')
def hello_world():
    return "Hello, World!"

    
@app.route('/predict/<prompt>', methods=['GET'])
def predict(prompt):
    try:
        if 'conversation_history' not in session:
            session['conversation_history'] = []

        history = session['conversation_history']

        response = predict_disease(history, prompt)

        # Assuming predict_disease returns the AI's response text
        session['conversation_history'].append({'user': prompt, 'ai': response})

        return jsonify(dict(response=response))
    except Exception as e:
        return jsonify({"error": str(e)}), 500
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)