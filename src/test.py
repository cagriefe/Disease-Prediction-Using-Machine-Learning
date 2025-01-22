import joblib
import pandas as pd
import numpy as np
from transformers import BioGptForCausalLM, BioGptTokenizer
import torch
from disease_prediction import DiseasePredictor
import os

# Define model paths
MODEL_PATH = '/Users/cagriefe/Git_pull/Disease-Prediction-Using-Machine-Learning/models'
TREATMENT_MODEL = os.path.join(MODEL_PATH, 'model.safetensors')

class IntegratedMedicalSystem:
    def __init__(self):
        # Initialize disease predictor
        self.predictor = DiseasePredictor()
        self.valid_symptoms = set(self.predictor.features)
        
        try:
            # Initialize treatment plan components from local model
            self.generator = BioGptForCausalLM.from_pretrained('microsoft/BioGPT')
            self.tokenizer = BioGptTokenizer.from_pretrained('microsoft/BioGPT')
            self.generator.eval()
        except Exception as e:
            print(f"Error loading treatment model: {str(e)}")
            raise

    def validate_symptoms(self, symptoms):
        """Validate and process input symptoms"""
        processed = []
        for symptom in symptoms.split(','):
            symptom = symptom.strip().lower().replace(' ', '_')
            if symptom in self.valid_symptoms:
                processed.append(symptom)
            else:
                print(f"Warning: Invalid symptom '{symptom}'")
        return processed

    def predict_and_treat(self, symptoms):
        try:
            # Validate symptoms
            valid_symptoms = self.validate_symptoms(symptoms)
            if not valid_symptoms:
                raise ValueError("No valid symptoms provided")
                
            # Get disease prediction
            prediction = self.predictor.return_data(valid_symptoms)
            
            # Generate treatment plan prompt
            prompt = f"""Generate a detailed treatment plan for a patient with {prediction['predicted_disease']}.
            Patient Symptoms: {', '.join(prediction['selected_symptoms'])}
            
            Please provide:
            1. Medications and dosage
            2. Lifestyle recommendations
            3. Follow-up care
            4. Warning signs
            
            Treatment Plan:"""
            
            # Generate treatment plan
            input_ids = self.tokenizer(prompt, return_tensors='pt').input_ids
            
            with torch.no_grad():
                outputs = self.generator.generate(
                    input_ids,
                    max_length=1000,
                    num_return_sequences=1,
                    temperature=0.7,
                    top_p=0.95,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            treatment_plan = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return {
                'disease': prediction['predicted_disease'],
                'symptoms': prediction['selected_symptoms'],
                'treatment_plan': treatment_plan.split('Treatment Plan:')[-1].strip()
            }
        except Exception as e:
            print(f"Error in predict_and_treat: {str(e)}")
            raise

def main():
    try:
        system = IntegratedMedicalSystem()
        
        # Test with known valid symptoms
        test_symptoms = [
            "itching,skin_rash,nodal_skin_eruptions",
            "continuous_sneezing,chills,fatigue"
        ]
        
        for symptoms in test_symptoms:
            print(f"\nTesting with symptoms: {symptoms}")
            result = system.predict_and_treat(symptoms)
            print(f"Predicted Disease: {result['disease']}")
            print(f"Symptoms: {', '.join(result['symptoms'])}")
            print(f"\nTreatment Plan:\n{result['treatment_plan']}")
            print("-" * 80)
            
    except Exception as e:
        print(f"Error in main execution: {str(e)}")

if __name__ == "__main__":
    main()