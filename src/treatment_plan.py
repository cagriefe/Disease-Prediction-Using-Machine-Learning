from transformers import BioGptForSequenceClassification, BioGptTokenizer, BioGptForCausalLM
import torch
import pandas as pd

def load_models():
    """Load both the classification model and generative model"""
    # Load classification model for disease prediction
    classifier = BioGptForSequenceClassification.from_pretrained('/Users/cagriefe/Git_pull/Disease-Prediction-Using-Machine-Learning/models/')
    # Load generative model for treatment plans
    generator = BioGptForCausalLM.from_pretrained('microsoft/BioGPT')
    # Load tokenizer
    tokenizer = BioGptTokenizer.from_pretrained('microsoft/BioGPT')
    # Load disease labels
    label_classes = pd.read_csv('label_classes.csv').iloc[:, 0].tolist()
    
    return classifier, generator, tokenizer, label_classes

def generate_treatment_plan(symptoms, classifier, generator, tokenizer, label_classes, max_length=512):
    """
    Generate a treatment plan based on symptoms:
    1. First predicts the disease using classifier
    2. Then generates treatment plan using the generative model
    """
    # First predict the disease
    classifier.eval()
    inputs = tokenizer(
        symptoms,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    with torch.no_grad():
        outputs = classifier(**inputs)
        predictions = torch.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(predictions, dim=1).item()
        predicted_disease = label_classes[predicted_class]
        confidence = predictions[0][predicted_class].item()
    
    # Generate treatment plan prompt
    prompt = f"Generate a detailed treatment plan for a patient with {predicted_disease}. Their symptoms are: {symptoms}\n\nTreatment Plan:"
    
    # Generate treatment plan
    generator.eval()
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids
    
    with torch.no_grad():
        outputs = generator.generate(
            input_ids,
            max_length=1000,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    treatment_plan = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return {
        'predicted_disease': predicted_disease,
        'confidence': f"{confidence:.2%}",
        'symptoms': symptoms,
        'treatment_plan': treatment_plan.split('Treatment Plan:')[-1].strip()
    }

def get_treatment_recommendation(symptoms):
    """Wrapper function for easy use"""
    classifier, generator, tokenizer, label_classes = load_models()
    return generate_treatment_plan(symptoms, classifier, generator, tokenizer, label_classes)

# Example usage
if __name__ == "__main__":
    # Example symptoms to test
    test_symptoms = [
        "Patient has severe headache with nausea and sensitivity to light",
        "Persistent cough with fever and fatigue for the past week"
    ]
    
    # Load models once
    classifier, generator, tokenizer, label_classes = load_models()
    
    # Test the treatment plan generation
    print("Generating treatment plans:\n")
    for symptoms in test_symptoms:
        print(f"Symptoms: {symptoms}")
        result = generate_treatment_plan(symptoms, classifier, generator, tokenizer, label_classes)
        print(f"\nPredicted Disease: {result['predicted_disease']}")
        print(f"Confidence: {result['confidence']}")
        print("\nRecommended Treatment Plan:")
        print(result['treatment_plan'])
        print("\n" + "="*50 + "\n")