from flask import Flask, render_template, request
from model import load_model
from utils.preprocess import preprocess_input

app = Flask(__name__)
model = load_model()


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    prediction_class = None
    confidence = None
    explanation = None
    action = None

    if request.method == "POST":
        usage_hours = float(request.form["usage_hours"])
        complaints = float(request.form["complaints"])

        input_data = preprocess_input(usage_hours, complaints)

        result = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        confidence = round(probability * 100, 1)

        if result == 1:
            prediction = "High Churn Risk"
            prediction_class = "fail"
            explanation = (
                "Low usage combined with frequent complaints indicates "
                "customer disengagement patterns associated with churn."
            )
            action = "Immediate retention action recommended."
        else:
            prediction = "Low Churn Risk"
            prediction_class = "pass"
            explanation = (
                "Consistent usage and minimal complaints indicate "
                "strong engagement and retention likelihood."
            )
            action = "No immediate action required. Continue monitoring."

    return render_template(
        "index.html",
        prediction=prediction,
        prediction_class=prediction_class,
        confidence=confidence,
        explanation=explanation,
        action=action
    )


if __name__ == "__main__":
    app.run(debug=True)
