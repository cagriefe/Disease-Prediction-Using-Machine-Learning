from flask import Flask, render_template, request, jsonify
from src.disease_prediction import DiseasePredictor

app = Flask(__name__)

# Initialize the DiseasePredictor
predictor = DiseasePredictor()

@app.route('/')
def home():
    symptoms = predictor.features
    return render_template('index.html', symptoms=symptoms)

@app.route('/predict', methods=['POST'])
def predict():
    selected_symptoms = request.form.getlist('symptoms')
    input_vector = predictor.create_input_vector(selected_symptoms)
    predicted_disease = predictor.predict_disease(input_vector)
    return jsonify({'predicted_disease': predicted_disease, 'selected_symptoms': selected_symptoms})

if __name__ == '__main__':
    app.run(debug=True)