import joblib
import pandas as pd
import numpy as np
from transformers import BioGptForCausalLM, BioGptTokenizer
import torch
from sklearn.preprocessing import LabelEncoder

class IntegratedMedicalSystem:
    def __init__(self, model_path='models/voting_classifier.pkl', 
                 training_data_path='data/disease_prediction/training.csv'):
        # Previous initialization code remains the same
        self.data = pd.read_csv(training_data_path).dropna(axis=1)
        self.features = self.data.columns[:-1]
        self.model = joblib.load(model_path)
        self.encoder = LabelEncoder()
        self.encoder.fit(self.data['prognosis'])
        
        # Create symptom index dictionary
        self.symptom_index = {}
        for index, value in enumerate(self.features):
            symptom = " ".join([i.capitalize() for i in value.split("_")])
            self.symptom_index[symptom] = index
        
        # Initialize treatment plan components
        self.generator = BioGptForCausalLM.from_pretrained('microsoft/BioGPT')
        self.tokenizer = BioGptTokenizer.from_pretrained('microsoft/BioGPT')
        self.generator.eval()

    # Previous methods remain the same until generate_treatment_plan
    def process_symptoms(self, symptoms):
        """Convert symptom string to binary feature vector"""
        input_vector = np.zeros(len(self.features))
        for symptom in symptoms.split(','):
            symptom = " ".join([i.capitalize() for i in symptom.strip().split()])
            if symptom in self.symptom_index:
                input_vector[self.symptom_index[symptom]] = 1
        return input_vector
    def predict_disease(self, symptoms):
        """Predict disease based on symptoms"""
        input_vector = self.process_symptoms(symptoms)
        prediction = self.model.predict([input_vector])
        disease = self.encoder.inverse_transform(prediction)[0]
        
        return {
            'disease': disease,
            'symptoms': [s.strip() for s in symptoms.split(',')]
        }

    def get_complete_diagnosis(self, symptoms):
        """Get complete diagnosis including disease prediction and treatment plan"""
        # Get disease prediction
        prediction_result = self.predict_disease(symptoms)
        
        # Generate treatment plan
        complete_result = self.generate_treatment_plan(prediction_result)
        
        return complete_result
        
    def generate_treatment_plan(self, prediction_result):
        """Generate detailed treatment plan based on predicted disease and symptoms"""
        try:
            symptoms_text = ", ".join(prediction_result['symptoms'])
            
            # Enhanced prompt for more detailed treatment plan
            prompt = f"""As a medical professional, provide a comprehensive treatment plan for a patient diagnosed with {prediction_result['disease']}.

    Patient Symptoms: {symptoms_text}

    Please provide a detailed treatment plan including:
    1. Medications and dosage recommendations
    2. Lifestyle modifications and self-care measures
    3. Follow-up recommendations
    4. Preventive measures
    5. Warning signs to watch for

    Treatment Plan:"""
            
            input_ids = self.tokenizer(prompt, return_tensors='pt').input_ids
            
            with torch.no_grad():
                outputs = self.generator.generate(
                    input_ids,
                    max_length=2000,  # Increased length for more detailed response
                    num_return_sequences=1,
                    no_repeat_ngram_size=2,
                    temperature=0.8,  # Slightly increased for more detailed generation
                    top_p=0.95,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    length_penalty=1.5,  # Encourage longer generations
                    repetition_penalty=1.2  # Discourage repetitive text
                )
            
            treatment_plan = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up the treatment plan text
            if "Treatment Plan:" in treatment_plan:
                treatment_plan = treatment_plan.split("Treatment Plan:")[-1].strip()
            
            # If the treatment plan is too short or generic, provide a backup structured response
            if len(treatment_plan.split()) < 50 or "review" in treatment_plan.lower():
                treatment_plan = self.generate_backup_treatment_plan(prediction_result['disease'])
            
            return {
            'predicted_disease': prediction_result['disease'],
            'symptoms': prediction_result['symptoms'],
            'treatment_plan': treatment_plan
        }
        except Exception as e:
            print(f"Error in treatment plan generation: {str(e)}")
            raise

    def generate_backup_treatment_plan(self, disease):
        """Generate a structured backup treatment plan when BioGPT response is inadequate"""
        disease_treatments = {
            "Fungal infection": """Treatment Plan for Fungal Infection:

1. Medications:
   - Topical antifungal cream (e.g., clotrimazole, miconazole) - Apply 2-3 times daily to affected areas
   - For severe cases: Oral antifungal medication may be prescribed

2. Self-Care Measures:
   - Keep affected areas clean and dry
   - Use breathable fabrics and avoid tight clothing
   - Change clothes and undergarments daily
   - Use antifungal powders in skin folds if recommended

3. Lifestyle Modifications:
   - Maintain good personal hygiene
   - Avoid sharing personal items
   - Use separate towels for affected areas
   - Avoid scratching affected areas

4. Prevention:
   - Keep skin dry, especially after bathing
   - Avoid prolonged exposure to moisture
   - Use antifungal powder in prone areas
   - Wear loose-fitting, breathable clothing

5. Follow-up Care:
   - Monitor for 2-4 weeks
   - Seek medical attention if condition worsens
   - Complete full course of medication
   - Regular check-ups if condition is recurring

6. Warning Signs:
   - Spreading of infection
   - Increased redness or swelling
   - Development of fever
   - Secondary bacterial infection signs
   - Persistent symptoms despite treatment""",
            # Add more disease treatments as needed
        }
        
        return disease_treatments.get(disease, "Please consult a healthcare provider for a detailed treatment plan for your specific condition.")

    # Rest of the class implementation remains the same
def main():
    try:
        medical_system = IntegratedMedicalSystem()
        symptoms = "Itching,Skin Rash,Nodal Skin Eruptions"
        result = medical_system.get_complete_diagnosis(symptoms)
        
        print("\nDiagnosis and Treatment Plan:")
        print(f"Predicted Disease: {result['predicted_disease']}")
        print(f"Symptoms: {', '.join(result['symptoms'])}")
        print("\nRecommended Treatment Plan:")
        print(result['treatment_plan'])
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")

if __name__ == "__main__":
    main()