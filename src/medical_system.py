from transformers import BioGptForCausalLM, BioGptTokenizer
import torch
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

class IntegratedMedicalSystem:
    def __init__(self, voting_classifier_path='models/voting_classifier.pkl', 
                 training_data_path='/Users/cagriefe/Git_pull/Disease-Prediction-Using-Machine-Learning/data/disease_prediction/training.csv'):
        # Initialize Disease Predictor components
        self.data = pd.read_csv(training_data_path).dropna(axis=1)
        self.features = self.data.columns[:-1]
        self.model = joblib.load(voting_classifier_path)
        self.encoder = LabelEncoder()
        self.encoder.fit(self.data['prognosis'])
        
        # Initialize Treatment Plan components
        self.generator = BioGptForCausalLM.from_pretrained('microsoft/BioGPT')
        self.tokenizer = BioGptTokenizer.from_pretrained('microsoft/BioGPT')
        self.generator.eval()  # Set to evaluation mode
        
    def create_input_vector(self, selected_symptoms):
        """Convert selected symptoms to model input vector"""
        input_vector = np.zeros(len(self.features))
        for symptom in selected_symptoms:
            if symptom in self.features:
                index = self.features.get_loc(symptom)
                input_vector[index] = 1
        return input_vector.reshape(1, -1)
    
    def predict_disease(self, selected_symptoms):
        """Predict disease based on symptoms"""
        input_vector = self.create_input_vector(selected_symptoms)
        prediction = self.model.predict(input_vector)
        predicted_disease = self.encoder.inverse_transform(prediction)[0]
        
        # Get prediction probability
        probabilities = self.model.predict_proba(input_vector)
        confidence = np.max(probabilities)
        
        return {
            'disease': predicted_disease,
            'confidence': f"{confidence:.2%}",
            'symptoms': selected_symptoms
        }
    
    def generate_treatment_plan(self, prediction_result):
        """Generate treatment plan based on predicted disease and symptoms"""
        # Create prompt for treatment plan generation
        symptoms_text = ", ".join(prediction_result['symptoms'])
        prompt = f"Generate a detailed treatment plan for a patient with {prediction_result['disease']}. "\
                f"Their symptoms are: {symptoms_text}\n\nTreatment Plan:"
        
        # Generate treatment plan
        input_ids = self.tokenizer(prompt, return_tensors='pt').input_ids
        
        with torch.no_grad():
            outputs = self.generator.generate(
                input_ids,
                max_length=1000,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        treatment_plan = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return {
            'predicted_disease': prediction_result['disease'],
            'confidence': prediction_result['confidence'],
            'symptoms': prediction_result['symptoms'],
            'treatment_plan': treatment_plan.split('Treatment Plan:')[-1].strip()
        }
    
    def get_complete_diagnosis(self, symptoms):
        """Complete pipeline from symptoms to treatment plan"""
        # First get disease prediction
        prediction = self.predict_disease(symptoms)
        
        # Then generate treatment plan
        complete_result = self.generate_treatment_plan(prediction)
        
        return complete_result

def main():
    # Initialize the integrated system
    medical_system = IntegratedMedicalSystem()
    
    # Example symptoms
    test_symptoms = ['headache', 'nausea', 'sensitivity_to_light']
    
    # Get complete diagnosis
    result = medical_system.get_complete_diagnosis(test_symptoms)
    
    # Print results
    print("\nDiagnosis and Treatment Plan:")
    print(f"Predicted Disease: {result['predicted_disease']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Symptoms: {', '.join(result['symptoms'])}")
    print("\nRecommended Treatment Plan:")
    print(result['treatment_plan'])

if __name__ == "__main__":
    main()