import pickle
import pandas as pd
from flask import Flask, request, render_template, jsonify

# Load the trained pipeline
with open("D:\GitHub\Predicting Student Academic Performance\model\student_performance_pipeline.pkl", "rb") as f:
    model_pipeline = pickle.load(f)

# Columns expected by the pipeline
expected_cols = list(model_pipeline.feature_names_in_)

# Numeric features (must be cast to float)
numeric_features = [
    "age", "Medu", "Fedu", "traveltime", "studytime",
    "failures", "famrel", "freetime", "goout",
    "Dalc", "Walc", "health", "absences", "G1", "G2"
]

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect form input
        data = request.form.to_dict()
        df = pd.DataFrame([data])

        # Convert numeric columns to float
        for col in numeric_features:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Ensure correct column order for pipeline
        df = df.reindex(columns=expected_cols, fill_value=0)

        # Prediction
        prediction = model_pipeline.predict(df)[0]

        return render_template(
            "index.html",
            prediction_text=f"Predicted Final Grade (G3): {prediction:.2f}"
        )

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
