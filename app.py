import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

class DiseasePredictorApp:
    def __init__(self):
        # Load and preprocess data
        self.data = pd.read_csv('data/training.csv').dropna(axis=1)
        self.features = self.data.columns[:-1]
        
        # Load model and initialize encoder
        self.model = joblib.load('models/voting_classifier.pkl')
        self.encoder = LabelEncoder()
        self.encoder.fit(self.data['prognosis'])
        
    def get_user_input(self):
        print("\nAvailable symptoms:")
        for idx, symptom in enumerate(self.features, 1):
            print(f"{idx}. {symptom.replace('_', ' ').title()}")
        
        input_vector = np.zeros(len(self.features))
        
        while True:
            try:
                choice = input("\nEnter symptom number (0 to finish): ")
                if not choice.strip():
                    continue
                    
                choice = int(choice)
                if choice == 0:
                    break
                if 1 <= choice <= len(self.features):
                    input_vector[choice-1] = 1
                    print(f"Added: {self.features[choice-1].replace('_', ' ').title()}")
                else:
                    print("Invalid symptom number!")
            except ValueError:
                print("Please enter a valid number!")
        
        return input_vector

    def predict_disease(self, input_vector):
        input_vector = input_vector.reshape(1, -1)
        prediction = self.model.predict(input_vector)
        return self.encoder.inverse_transform(prediction)[0]

    def run(self):
        print("Disease Prediction System")
        print("========================")
        
        while True:
            input_vector = self.get_user_input()
            
            if np.any(input_vector):
                predicted_disease = self.predict_disease(input_vector)
                selected_symptoms = [s for i, s in enumerate(self.features) if input_vector[i] == 1]
                
                print("\nSelected symptoms:")
                for symptom in selected_symptoms:
                    print(f"- {symptom.replace('_', ' ').title()}")
                print(f"\nPredicted Disease: {predicted_disease}")
            
            if input("\nTry again? (y/n): ").lower() != 'y':
                break
        
        print("\nThank you for using the Disease Prediction System!")

if __name__ == "__main__":
    try:
        app = DiseasePredictorApp()
        app.run()
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")