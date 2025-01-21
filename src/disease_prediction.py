import pickle
import numpy as np

def load_model():
    with open('models/voting_classifier.pkl', 'rb') as f:
        return pickle.load(f)

def predict_disease(model, symptoms):
    # Convert symptoms to model input format (dummy implementation)
    input_vector = np.array([1 if symptom in symptoms else 0 for symptom in all_possible_symptoms])
    return model.predict([input_vector])[0]