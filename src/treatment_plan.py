from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import logging

# Setup logging for generation failures
logging.basicConfig(filename='generation_failures.log', level=logging.WARNING)

def load_generator():
    """Load the ContactDoctor Bio-Medical Llama model."""
    tokenizer = AutoTokenizer.from_pretrained("ContactDoctor/Bio-Medical-Llama-3-2-1B-CoT-012025", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("ContactDoctor/Bio-Medical-Llama-3-2-1B-CoT-012025", trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer

def generate_treatment_plan(symptoms, predicted_disease, generator, tokenizer, max_length=1024):
    """Generate a concise treatment plan using the Bio-Medical Llama model."""
    formatted_symptoms = ", ".join([symptom.capitalize() for symptom in symptoms])
    
    # Simplified and concise prompt
    prompt = f"""
    Provide a concise treatment plan for a patient diagnosed with {predicted_disease}. 
    Key symptoms: {formatted_symptoms}.
    
    Include:
    - One key medication and dosage
    - Essential lifestyle modifications
    - Follow-up advice
    - Preventive strategies
    - Warning signs to seek help
    
    Treatment Plan (1024 tokens max):
    """

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, padding=True, max_length=tokenizer.model_max_length)

    try:
        outputs = generator.generate(
            inputs.input_ids,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,  # Lower temperature for focused responses
            top_p=0.9,       # Diverse sampling
            num_beams=5,     # Enable beam search
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id
        )
        raw_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        treatment_plan = raw_output.split("Treatment Plan:")[-1].strip()

        # Debug generated output
        print(f"Raw model output: {raw_output}")

        # Validate response
        if len(treatment_plan.split()) < 20:  # Validate length to ensure minimal response
            logging.warning(f"Inadequate response for {predicted_disease}. Using fallback...")
            # treatment_plan = generate_dynamic_fallback(predicted_disease)

    except Exception as e:
        logging.error(f"Error during generation for {predicted_disease}: {e}")
        treatment_plan = "An error occurred. Please consult a healthcare provider."

    return {
        'predicted_disease': predicted_disease,
        'symptoms': formatted_symptoms,
        'treatment_plan': treatment_plan
    }


MODEL = "ContactDoctor/Bio-Medical-Llama-3-2-1B-CoT-012025"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
tokenizer.model_max_length