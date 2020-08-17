from flask import Flask, request, jsonify, render_template, url_for, redirect
import translate

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/translate_page", methods=['GET'])
def translate_page():
    english_sentence = request.args.get('english')
    french_sentence = translate.translate_to_french(english_sentence)
    return jsonify(french_sentence)

if __name__ == "__main__":
    app.run(debug=True)