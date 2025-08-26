import argparse
import numpy as np
import os
import torch
import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import BartForConditionalGeneration, BertTokenizer, get_linear_schedule_with_warmup

"""
Mandarin-to-Cantonese Neural Machine Translation (NMT) Project

This script fine-tunes a pre-trained BART (Bidirectional and Auto-Regressive Transformer)
model for the task of translating Mandarin Chinese to Cantonese. It includes functions for
data loading, model training, translation, and evaluation using the BLEU score.

Key Libraries:
- PyTorch
- Hugging Face Transformers (for BART model and tokenizer)
- NLTK (for BLEU score calculation)
"""

# --- 1. Configuration and Setup ---

# Use GPU if available, otherwise fall back to CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Initialize tokenizer from a pre-trained Chinese BART model
TOKENIZER = BertTokenizer.from_pretrained("fnlp/bart-base-chinese")

class TranslationDataset(Dataset):
    """Custom PyTorch Dataset for handling translation data."""
    def __init__(self, data, tokenizer, max_length=30):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, index):
        source, target = self.data[index]
        
        # Tokenize source (Mandarin) sentence
        input_ids = self.tokenizer.encode_plus(
            source,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )["input_ids"]
        
        # Tokenize target (Cantonese) sentence
        target_ids = self.tokenizer.encode_plus(
            target,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )["input_ids"]

        return input_ids.squeeze(), target_ids.squeeze()

    def __len__(self):
        return len(self.data)

# --- 2. Core Functions ---

def load_data(file_path):
    """Loads parallel data from a tab-separated text file."""
    x_data, y_data = [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                x_data.append(parts[0])
                y_data.append(parts[1])
    return x_data, y_data

def train_model(model, data_loader, optimizer, scheduler, num_epochs, save_path):
    """Fine-tunes the BART model on the provided dataset."""
    model.train()
    print("Starting model training...")

    for epoch in range(num_epochs):
        print("-" * 50)
        print(f'Epoch {epoch + 1}/{num_epochs}')
        total_loss = 0

        for batch in tqdm.tqdm(data_loader, desc=f"Training Epoch {epoch + 1}"):
            input_ids, target_ids = [b.to(DEVICE) for b in batch]
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, labels=target_ids)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(data_loader)
        print(f'Average Training Loss: {avg_train_loss:.4f}')

    # Save the fine-tuned model
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    model.save_pretrained(save_path)
    print(f'Model successfully saved to {save_path}')
    print("-" * 50)

def translate_sentence(sentence, model, tokenizer, device, max_length=30):
    """Generates a translation for a single input sentence."""
    model.eval()
    model = model.to(device)

    tokens = tokenizer.encode_plus(
        sentence,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=4,
            early_stopping=True
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def evaluate_model(model, test_data_path, tokenizer, device):
    """Evaluates the model on a test set and computes the average BLEU score."""
    print(f"\nEvaluating model on: {test_data_path}")
    x_test, y_test = load_data(test_data_path)
    bleu_scores = []
    smoothing_func = SmoothingFunction().method4

    for source, reference in tqdm.tqdm(zip(x_test, y_test), total=len(x_test), desc="Evaluating"):
        translation = translate_sentence(source, model, tokenizer, device)
        # NLTK expects tokenized sentences (list of characters for Chinese)
        score = sentence_bleu([list(reference)], list(translation), smoothing_function=smoothing_func)
        bleu_scores.append(score)
        
    avg_bleu = np.mean(bleu_scores)
    print(f"Average BLEU score on {len(x_test)} sentences: {avg_bleu:.4f}")
    return avg_bleu

# --- 3. Main Execution Block ---

def main(args):
    """Main function to orchestrate the training and evaluation process."""
    
    # Load the base pre-trained model
    model = BartForConditionalGeneration.from_pretrained("fnlp/bart-base-chinese")
    model = model.to(DEVICE)

    if args.mode == 'train':
        # --- Training Mode ---
        x_train, y_train = load_data(args.train_file)
        train_dataset = TranslationDataset(list(zip(x_train, y_train)), TOKENIZER)
        data_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)

        optimizer = AdamW(model.parameters(), lr=args.learning_rate)
        total_steps = len(data_loader) * args.epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        train_model(model, data_loader, optimizer, scheduler, args.epochs, args.model_save_path)
    
    elif args.mode == 'evaluate':
        # --- Evaluation Mode ---
        print(f"Loading fine-tuned model from: {args.model_load_path}")
        model = BartForConditionalGeneration.from_pretrained(args.model_load_path)
        evaluate_model(model, args.test_file, TOKENIZER, DEVICE)
        
    elif args.mode == 'translate':
        # --- Interactive Translation Mode ---
        print(f"Loading fine-tuned model from: {args.model_load_path}")
        model = BartForConditionalGeneration.from_pretrained(args.model_load_path)
        while True:
            sentence = input("Enter a Mandarin sentence to translate (or 'quit' to exit): ")
            if sentence.lower() == 'quit':
                break
            translation = translate_sentence(sentence, model, TOKENIZER, DEVICE)
            print(f"Cantonese Translation: {translation}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Mandarin-to-Cantonese Neural Machine Translation")
    
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'evaluate', 'translate'],
                        help="Operation mode: train, evaluate, or translate.")
    
    # Training arguments
    parser.add_argument('--train_file', type=str, default='data/training_900.txt', help="Path to the training data file.")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs.")
    parser.add_argument('--learning_rate', type=float, default=2e-5, help="Learning rate for AdamW optimizer.")
    parser.add_argument('--batch_size', type=int, default=16, help="Training batch size.")
    parser.add_argument('--model_save_path', type=str, default='./models/mandarin_to_cantonese_bart', help="Directory to save the fine-tuned model.")

    # Evaluation/Translation arguments
    parser.add_argument('--test_file', type=str, default='data/test_100.txt', help="Path to the test data file for evaluation.")
    parser.add_argument('--model_load_path', type=str, default='./models/mandarin_to_cantonese_bart', help="Directory to load the fine-tuned model from.")

    args = parser.parse_args()
    main(args)
