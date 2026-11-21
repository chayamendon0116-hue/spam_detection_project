
from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("spam_model.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    result = ""
    message = ""

    if request.method == "POST":
        message = request.form["message"].strip()

        if message:
            prediction = model.predict([message])[0]

            if prediction == "spam":
                result = "SPAM"
            else:
                result = "NOT SPAM"

    return render_template(
        "index.html",
        result=result,
        message=message
    )

if __name__ == "__main__":
    app.run(debug=False)
