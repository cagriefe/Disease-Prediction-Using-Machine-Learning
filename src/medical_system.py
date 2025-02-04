import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='urllib3')
import joblib
from .disease_prediction import DiseasePredictor
from .treatment_plan import load_generator, generate_treatment_plan
import pandas as pd

class MedicalSystem:
    def __init__(self):
        # Initialize the disease predictor and the Medichat generator
        self.disease_predictor = DiseasePredictor()
        self.generator, self.tokenizer = load_generator()  # Use the Medichat-Llama3-8B model
        # Load and preprocess data
        self.data = pd.read_csv('data/disease_prediction/training.csv').dropna(axis=1)
        self.features = self.data.columns[:-1]

    def process_case(self, symptoms_list):
        # Predict disease
        prediction_result = self.disease_predictor.return_data(symptoms_list)
        predicted_disease = prediction_result['predicted_disease']

        # Format symptoms for treatment plan generation
        formatted_symptoms = [symptom.lower() for symptom in symptoms_list]

        # Generate treatment plan ONCE
        treatment_result = generate_treatment_plan(
            symptoms=formatted_symptoms,
            predicted_disease=predicted_disease,
            generator=self.generator,
            tokenizer=self.tokenizer,
            max_length=1024  # Adjusted max_length for Medichat model
        )

        return {
            'symptoms': symptoms_list,
            'formatted_symptoms': ", ".join(formatted_symptoms),
            'predicted_disease': predicted_disease,
            'treatment_plan': treatment_result['treatment_plan']
        }



# if __name__ == "__main__":
#     system = MedicalSystem()
    
#     # Example usage
#     # test_symptoms = ["itching", "skin_rash", "nodal_skin_eruptions"]
#     # test_symptoms = ["continuous_sneezing", "shivering", "chills"]
#     test_symptoms = ["stomach_pain", "acidity", "ulcers_on_tongue", "vomiting"]
#     # test_symptoms = ["fatigue", "weight_gain", "anxiety", "cold_hands_and_feets"]
#     # test_symptoms = ["weight_loss", "restlessness", "lethargy", "patches_in_throat"]
#     print("Medical Diagnosis and Treatment System\n")
#     print("Input Symptoms:", test_symptoms)
    
#     result = system.process_case(test_symptoms)
    
#     print("\nDiagnosis Results:")
#     print(f"- Predicted Disease: {result['predicted_disease']}")
#     print(f"- Identified Symptoms: {result['formatted_symptoms']}")
#     print("\nRecommended Treatment Plan:")
#     print(result['treatment_plan'])