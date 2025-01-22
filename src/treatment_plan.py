from transformers import BioGptForCausalLM, BioGptTokenizer
import torch

def load_generator(model_dir='/Users/cagriefe/Git_pull/Disease-Prediction-Using-Machine-Learning/models'):
    """Load the fine-tuned treatment generation model"""
    tokenizer = BioGptTokenizer.from_pretrained('microsoft/BioGPT')
    generator = BioGptForCausalLM.from_pretrained(model_dir)
    return generator, tokenizer

def generate_treatment_plan(symptoms, predicted_disease, generator, tokenizer, max_length=2000):
    """Generate treatment plan using provided disease and symptoms"""
    # Create prompt with predicted disease
    prompt = f"""As a medical professional, provide a comprehensive treatment plan for a patient diagnosed with {predicted_disease}.

Patient Symptoms: {symptoms}

Please provide a detailed treatment plan including:
1. Medications and dosage recommendations
2. Lifestyle modifications and self-care measures
3. Follow-up recommendations
4. Preventive measures
5. Warning signs to watch for

Treatment Plan:"""
    
    # Ensure prompt length does not exceed model's maximum length
    inputs = tokenizer(prompt, return_tensors='pt', max_length=max_length, truncation=True)
    
    with torch.no_grad():
        outputs = generator.generate(
            inputs.input_ids,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the treatment plan part
    treatment_plan = full_response.split("Treatment Plan:")[-1].strip()
    
    # If the treatment plan is too short or generic, provide a backup structured response
    if len(treatment_plan.split()) < 50 or "review" in treatment_plan.lower():
        treatment_plan = generate_backup_treatment_plan(predicted_disease)
    
    return {
        'predicted_disease': predicted_disease,
        'symptoms': symptoms,
        'treatment_plan': treatment_plan
    }

def generate_backup_treatment_plan(disease):
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