# treatment_plan.py (corrected)
from transformers import BioGptForCausalLM, BioGptTokenizer
import torch

def load_generator():
    """Load the fine-tuned treatment generation model"""
    model_path = 'models/model.safetensors'  # Update with actual path
    tokenizer = BioGptTokenizer.from_pretrained('microsoft/BioGPT')
    generator = BioGptForCausalLM.from_pretrained(model_path)
    return generator, tokenizer

def generate_treatment_plan(symptoms, predicted_disease, generator, tokenizer, max_length=1000):
    """Generate treatment plan using provided disease and symptoms"""
    # Create prompt with predicted disease
    prompt = f"Patient diagnosed with {predicted_disease}. Symptoms: {symptoms}\nGenerate comprehensive treatment plan:\n"
    
    # Generate response
    inputs = tokenizer(prompt, return_tensors='pt', max_length=max_length, truncation=True)
    
    with torch.no_grad():
        outputs = generator.generate(
            inputs.input_ids,
            max_length=1500,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the treatment plan part
    treatment_plan = full_response.split("Generate comprehensive treatment plan:")[-1].strip()
    
    return {
        'predicted_disease': predicted_disease,
        'symptoms': symptoms,
        'treatment_plan': treatment_plan
    }