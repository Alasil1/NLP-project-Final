import os
import torch
from flask import Flask, request, jsonify
from transformers import BartTokenizer, BartForConditionalGeneration

app = Flask(__name__)

tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
model = BartForConditionalGeneration.from_pretrained("bart-samsum")
model.to("cuda" if torch.cuda.is_available() else "cpu")

@app.route("/summarize", methods=["POST"])
def summarize():
    try:
        data = request.get_json()
        dialogue = data.get("dialogue")
        if not dialogue or not isinstance(dialogue, str):
            return jsonify({"error": "Invalid or missing dialogue"}), 400
        inputs = tokenizer(dialogue, return_tensors="pt", max_length=1024, truncation=True).to(model.device)
        summary_ids = model.generate(inputs["input_ids"], max_length=50, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return jsonify({"summary": summary})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def index():
    return "Dialogue Summarization API is running. Use POST /summarize with JSON data."

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)