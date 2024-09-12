from flask import Flask

app = Flask(__name__)

@app.route("/")
def home():
    return "Generative AI Model for Text Summarization"

if __name__ == "__main__":
    app.run(debug=True)