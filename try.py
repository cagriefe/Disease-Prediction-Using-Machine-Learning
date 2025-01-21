from transformers import BioGptForSequenceClassification, BioGptTokenizer, Trainer, TrainingArguments
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset
import numpy as np

class MedicalDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.text = dataframe['text'].values
        self.targets = dataframe['disease'].values
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        target = self.targets[index]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(target, dtype=torch.long)
        }

def prepare_data(csv_path):
    # Load and preprocess the dataset
    df = pd.read_csv(csv_path)
    
    # Combine label and answer, handle missing values
    df['text'] = df['label'] + ' ' + df['answer']
    df = df[['text', 'disease']].dropna()
    
    # Encode disease labels
    label_encoder = LabelEncoder()
    df['disease'] = label_encoder.fit_transform(df['disease'])
    
    # Print dataset statistics
    print(f"Total samples: {len(df)}")
    print(f"Unique diseases: {len(df['disease'].unique())}")
    print("\nClass distribution:")
    print(df['disease'].value_counts())
    
    return df, label_encoder

def train_model(df, tokenizer, num_labels, max_len=512, batch_size=8, epochs=3):
    # Split dataset
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['disease'])
    
    # Create datasets
    train_dataset = MedicalDataset(train_df, tokenizer, max_len)
    val_dataset = MedicalDataset(val_df, tokenizer, max_len)
    
    # Initialize model with correct number of labels
    model = BioGptForSequenceClassification.from_pretrained(
        'microsoft/BioGPT',
        num_labels=num_labels,
        problem_type="single_label_classification"
    )
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss"
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    
    # Train model
    print("Starting training...")
    trainer.train()
    
    return trainer, model

def main():
    # Load tokenizer
    tokenizer = BioGptTokenizer.from_pretrained('microsoft/BioGPT')
    
    # Prepare data
    df, label_encoder = prepare_data('healifyLLM_answer_dataset.csv')
    
    # Train model
    trainer, model = train_model(
        df=df,
        tokenizer=tokenizer,
        num_labels=len(label_encoder.classes_),
        max_len=512,
        batch_size=8,
        epochs=3
    )
    
    # Save the trained model and label encoder
    model.save_pretrained('./medical_model')
    pd.Series(label_encoder.classes_).to_csv('label_classes.csv', index=False)

if __name__ == "__main__":
    main()