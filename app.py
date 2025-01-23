from flask import Flask, render_template, request, jsonify
from src.medical_system import MedicalSystem

app = Flask(__name__)

# Initialize the Predictor
predictor = MedicalSystem()

@app.route('/')
def home():
    symptoms = predictor.features
    return render_template('index.html', symptoms=symptoms)

@app.route('/predict', methods=['POST'])
def predict():
    selected_symptoms = request.form.getlist('symptoms')
    prediction = predictor.process_case(selected_symptoms)
    return jsonify(prediction)

if __name__ == '__main__':
    app.run(debug=True)