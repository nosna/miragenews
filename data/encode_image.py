import argparse
import torch
import os
from PIL import Image
from tqdm import tqdm
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from datasets import load_dataset

def preprocess_image(image_path, processor):
    image = processor(Image.open(image_path).convert("RGB")).data['pixel_values'][0]
    return torch.Tensor(image)

def preprocess_hf_image(hf_image, processor):
    # Process an image from the HF dataset (assuming PIL format)
    image = processor(hf_image.convert("RGB")).data['pixel_values'][0]
    return torch.Tensor(image)

def process_batch(images, model, device):
    images_tensor = torch.stack(images).to(device)
    with torch.no_grad():
        return model(images_tensor).pooler_output

def process_directory(directory, model, processor, device, batch_size=32):
    encodings = []
    batch = []
    for filename in tqdm(sorted(os.listdir(directory))):
        image_path = os.path.join(directory, filename)
        batch.append(preprocess_image(image_path, processor))
        if len(batch) == batch_size:
            encodings.append(process_batch(batch, model, device))
            batch = []

    if batch:
        encodings.append(process_batch(batch, model, device))

    return torch.cat(encodings)

def process_hf_dataset_by_label(dataset, model, processor, device, batch_size=32):
    encodings_real, encodings_fake = [], []
    batch_real, batch_fake = [], []
    for item in tqdm(dataset):
        image_tensor = preprocess_hf_image(item["image"], processor)
        label = item["label"]

        if label == 0:
            batch_real.append(image_tensor)
            if len(batch_real) == batch_size:
                encodings_real.append(process_batch(batch_real, model, device))
                batch_real = []
        elif label == 1:
            batch_fake.append(image_tensor)
            if len(batch_fake) == batch_size:
                encodings_fake.append(process_batch(batch_fake, model, device))
                batch_fake = []

    if batch_real:
        encodings_real.append(process_batch(batch_real, model, device))
    if batch_fake:
        encodings_fake.append(process_batch(batch_fake, model, device))

    return torch.cat(encodings_real), torch.cat(encodings_fake)

def save_encodings(encodings, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(encodings, filename)

def main(custom=False, read_dirs=None, batch_size=64):
    device = "cuda"
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl").vision_model.to(device)

    if custom and read_dirs:
        for read_dir in read_dirs:
            real_dir = os.path.join("my_dataset/image", read_dir, "real")
            fake_dir = os.path.join("my_dataset/image", read_dir, "fake")

            if os.path.exists(real_dir):
                output_file_real = f"encodings/image/{read_dir}/real.pt"
                image_features_real = process_directory(real_dir, model, processor, device, batch_size)
                save_encodings(image_features_real, output_file_real)
                print(f"Encoded features for real images in {read_dir} saved to {output_file_real}")

            if os.path.exists(fake_dir):
                output_file_fake = f"encodings/image/{read_dir}/fake.pt"
                image_features_fake = process_directory(fake_dir, model, processor, device, batch_size)
                save_encodings(image_features_fake, output_file_fake)
                print(f"Encoded features for fake images in {read_dir} saved to {output_file_fake}")

    else:
        dataset_name = "anson-huang/mirage-news"
        available_splits = load_dataset(dataset_name).keys()  # Get available splits from the dataset
        for split in available_splits:
            dataset = load_dataset(dataset_name, split=split)
            output_file_real = f"encodings/image/{split}/real.pt"
            output_file_fake = f"encodings/image/{split}/fake.pt"

            image_features_real, image_features_fake = process_hf_dataset_by_label(dataset, model, processor, device, batch_size)
            save_encodings(image_features_real, output_file_real)
            save_encodings(image_features_fake, output_file_fake)

            print(f"Encoded features for real images in {split} saved to {output_file_real}")
            print(f"Encoded features for fake images in {split} saved to {output_file_fake}")

    print("Feature vectors saved successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode images from local directory or HF dataset")
    parser.add_argument("--custom", action="store_true", help="Use local directories instead of Hugging Face dataset")
    parser.add_argument("--read_dirs", nargs="+", help="List of directories to read images from (if --custom is set)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for processing images (default: 64)")

    args = parser.parse_args()
    main(custom=args.custom, read_dirs=args.read_dirs, batch_size=args.batch_size)
