import argparse
import torch
import os
import clip
from tqdm import tqdm
from datasets import load_dataset

def preprocess_texts(texts, model, device):
    """Tokenizes and encodes a batch of text using CLIP's model."""
    tokenized_texts = clip.tokenize(texts, truncate=True).to(device)
    with torch.no_grad():
        return model.encode_text(tokenized_texts)

def process_custom_text_file(text_file, model, device, batch_size=64):
    """Reads texts from a custom text file and encodes them in batches."""
    encodings = []
    batch = []
    
    with open(text_file, 'r') as file:
        for line in tqdm(file, desc="Processing custom text file"):
            batch.append(line.strip())
            if len(batch) == batch_size:
                encodings.append(preprocess_texts(batch, model, device))
                batch = []
    
    if batch:
        encodings.append(preprocess_texts(batch, model, device))
    
    return torch.cat(encodings)

def process_hf_dataset_by_label(dataset, model, device, batch_size=64):
    """Encodes texts from a Hugging Face dataset based on labels in batches."""
    encodings_real, encodings_fake = [], []
    batch_real, batch_fake = [], []

    for item in tqdm(dataset, desc="Processing Hugging Face dataset"):
        text = item['text']
        label = item["label"]

        if label == 0:
            batch_real.append(text)
            if len(batch_real) == batch_size:
                encodings_real.append(preprocess_texts(batch_real, model, device))
                batch_real = []
        elif label == 1:
            batch_fake.append(text)
            if len(batch_fake) == batch_size:
                encodings_fake.append(preprocess_texts(batch_fake, model, device))
                batch_fake = []

    if batch_real:
        encodings_real.append(preprocess_texts(batch_real, model, device))
    if batch_fake:
        encodings_fake.append(preprocess_texts(batch_fake, model, device))

    return torch.cat(encodings_real), torch.cat(encodings_fake)

def save_encodings(encodings, filename):
    """Saves the encoded tensors to a file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(encodings, filename)

def main(custom=False, text_files=None, batch_size=64):
    device = "cuda"
    model, _ = clip.load("ViT-L/14@336px", device=device)

    if custom and text_files:
        for text_file in text_files:
            real_file = os.path.join("my_dataset/text", text_file, "real.txt")
            fake_file = os.path.join("my_dataset/text", text_file, "fake.txt")

            if os.path.exists(real_file):
                output_file_real = f"encodings/text/{text_file}/real.pt"
                text_features_real = process_custom_text_file(real_file, model, device, batch_size)
                save_encodings(text_features_real, output_file_real)
                print(f"Encoded features for real texts in {text_file} saved to {output_file_real}")

            if os.path.exists(fake_file):
                output_file_fake = f"encodings/text/{text_file}/fake.pt"
                text_features_fake = process_custom_text_file(fake_file, model, device, batch_size)
                save_encodings(text_features_fake, output_file_fake)
                print(f"Encoded features for fake texts in {text_file} saved to {output_file_fake}")

    else:
        dataset_name = "anson-huang/mirage-news"
        available_splits = load_dataset(dataset_name).keys()
        
        for split in available_splits:
            dataset = load_dataset(dataset_name, split=split)
            output_file_real = f"encodings/text/{split}/real.pt"
            output_file_fake = f"encodings/text/{split}/fake.pt"

            text_features_real, text_features_fake = process_hf_dataset_by_label(dataset, model, device, batch_size)
            save_encodings(text_features_real, output_file_real)
            save_encodings(text_features_fake, output_file_fake)

            print(f"Encoded features for real texts in {split} saved to {output_file_real}")
            print(f"Encoded features for fake texts in {split} saved to {output_file_fake}")

    print("Text feature vectors saved successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode text from local text files or HF dataset")
    parser.add_argument("--custom", action="store_true", help="Use local text files instead of Hugging Face dataset")
    parser.add_argument("--text_files", nargs="+", help="List of text files to read (if --custom is set)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for processing texts (default: 64)")

    args = parser.parse_args()
    main(custom=args.custom, text_files=args.text_files, batch_size=args.batch_size)
