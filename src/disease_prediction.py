import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

class DiseasePredictor:
    def __init__(self):
        # Load and preprocess data
        self.data = pd.read_csv('data/disease_prediction/training.csv').dropna(axis=1)
        self.features = self.data.columns[:-1]
        
        # Load model and initialize encoder
        self.model = joblib.load('models/voting_classifier.pkl')
        self.encoder = LabelEncoder()
        self.encoder.fit(self.data['prognosis'])
    
    def create_input_vector(self, selected_symptoms):
        input_vector = np.zeros(len(self.features))
        for symptom in selected_symptoms:
            if symptom in self.features:
                index = self.features.get_loc(symptom)
                input_vector[index] = 1
        return input_vector.reshape(1, -1)
    
    def predict_disease(self, input_vector):
        prediction = self.model.predict(input_vector)
        return self.encoder.inverse_transform(prediction)[0]
    
    def return_data(self, selected_symptoms):
        input_vector = self.create_input_vector(selected_symptoms)
        predicted_disease = self.predict_disease(input_vector)
        return {
            'selected_symptoms': selected_symptoms,
            'predicted_disease': predicted_disease
        }