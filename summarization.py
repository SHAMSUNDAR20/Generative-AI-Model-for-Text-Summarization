import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from flask import Flask, request, jsonify

# Initialize the model and tokenizer for summarization
model_name = 'facebook/bart-large-cnn'
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Initialize Flask app
app = Flask(__name__)

# Summarization function
def summarize_text(text):
    inputs = tokenizer([text], max_length=1024, return_tensors="pt", truncation=True)
    summary_ids = model.generate(inputs["input_ids"], max_length=150, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# API for summarization
@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.json
    input_text = data.get('text')
    summary = summarize_text(input_text)
    return jsonify({'summary': summary})

if __name__ == "__main__":
    app.run(debug=True)
