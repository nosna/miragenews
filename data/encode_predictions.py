import argparse
import torch
from PIL import Image
from tqdm import tqdm
from transformers import Owlv2Processor, Owlv2ForObjectDetection, Blip2Processor, Blip2ForConditionalGeneration
import clip
from datasets import load_dataset
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models import get_model
from data import load_config, get_object_class_caption, get_preprocessed_image
from utils import load_model_checkpoint

def get_logits(y, eps=1e-5):
    y = torch.clamp(y, eps, 1 - eps)
    y = torch.log(y / (1 - y))
    return y

def process_img_linear(model, batch, image_encoder, device):
    """Process images with the img-linear model."""
    batch_tensor = torch.stack(batch).to(device)
    with torch.no_grad():
        images_encoding = image_encoder(batch_tensor).pooler_output
        outputs = model(images_encoding)
    return get_logits(outputs)

def process_cbm_encoder(model, image, objects, object_processor, object_detector, image_processor, image_encoder, device):
    """Process each image with cbm-encoder model for object crops."""
    object_scores = torch.full((300,), float('-inf')).to(device)  # Initialize scores with -inf
    filled_indices = torch.zeros((300,), dtype=torch.bool).to(device)  # Track updated indices

    inputs = object_processor(text=objects, images=image, return_tensors="pt").to(device)
    unnormalized_image = get_preprocessed_image(inputs.pixel_values.cpu())
    outputs = object_detector(**inputs)
    target_sizes = torch.Tensor([unnormalized_image.size[::-1]])
    detected_objects = object_processor.post_process_object_detection(outputs=outputs, threshold=0.3, target_sizes=target_sizes)[0]

    for box, label in zip(detected_objects["boxes"], detected_objects["labels"]):
        obj_class_idx = label.item()  # Object class index
        crop_img = unnormalized_image.crop(box.tolist())
        crop_tensor = torch.Tensor(image_processor(crop_img.convert("RGB")).data['pixel_values'][0]).unsqueeze(0).to(device)
        crop_encoding = image_encoder(crop_tensor).pooler_output[0]
        with torch.no_grad():
            crop_score = model.classifiers[obj_class_idx](crop_encoding)
            object_scores[obj_class_idx] = torch.maximum(object_scores[obj_class_idx], crop_score)
            filled_indices[obj_class_idx] = True

    # Fill unfilled indices with 0.5
    object_scores[~filled_indices] = 0.5
    return get_logits(object_scores).unsqueeze(0)

def preprocess_texts(texts, model, device):
    """Tokenizes and encodes a batch of text using CLIP's model."""
    tokenized_texts = clip.tokenize(texts, truncate=True).to(device)
    with torch.no_grad():
        return model.encode_text(tokenized_texts)
    
def process_txt_linear(model, text_encoding, device):
    """Process images with the txt-linear model."""
    with torch.no_grad():
        outputs = model(text_encoding.float().to(device))
    return get_logits(outputs)

def save_predictions(predictions, output_dir, mode, model_class, split, label):
    output_path = os.path.join(output_dir, mode, model_class, split, f"{label}.pt")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(predictions, output_path)

def main(mode, model_class, custom=False, img_dirs=None, text_dirs=None, batch_size=64, test_only=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = "encodings/predictions"

    # Merge predictions
    if model_class == "merged":
        if mode == "image":
            linear_pred_dir = f'{output_dir}/{mode}/linear'
            cbm_pred_dir = f'{output_dir}/{mode}/cbm-encoder'
            
            for split in sorted(os.listdir(linear_pred_dir)):
                linear_real = torch.load(f'{linear_pred_dir}/{split}/real.pt').to(device)
                linear_fake = torch.load(f'{linear_pred_dir}/{split}/fake.pt').to(device)
                cbm_real = torch.load(f'{cbm_pred_dir}/{split}/real.pt').to(device)
                cbm_fake = torch.load(f'{cbm_pred_dir}/{split}/fake.pt').to(device)
                merged_dir = f'{output_dir}/{mode}/merged/{split}'
                os.makedirs(merged_dir, exist_ok=True)
                torch.save(torch.concat((cbm_real, linear_real), dim=1), f'{merged_dir}/real.pt')
                torch.save(torch.concat((cbm_fake, linear_fake), dim=1), f'{merged_dir}/fake.pt')
            print(f"Predictions merged.")
            return
        elif mode == "text":
            linear_pred_dir = f'{output_dir}/{mode}/linear'
            tbm_pred_dir = f'{output_dir}/{mode}/tbm-encoder'
            
            for split in sorted(os.listdir(linear_pred_dir)):
                linear_real = torch.load(f'{linear_pred_dir}/{split}/real.pt').to(device)
                linear_fake = torch.load(f'{linear_pred_dir}/{split}/fake.pt').to(device)
                tbm_real = torch.load(f'{tbm_pred_dir}/{split}/real.pt').to(device)
                tbm_fake = torch.load(f'{tbm_pred_dir}/{split}/fake.pt').to(device)
                print(linear_real.shape)
                print(tbm_real.shape)
                merged_dir = f'{output_dir}/{mode}/merged/{split}'
                os.makedirs(merged_dir, exist_ok=True)
                torch.save(torch.concat((tbm_real, linear_real), dim=1), f'{merged_dir}/real.pt')
                torch.save(torch.concat((tbm_fake, linear_fake), dim=1), f'{merged_dir}/fake.pt')
            print(f"Predictions merged.")
            return
    
    if mode == "image":
        # Load shared processors and models
        object_processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
        object_detector = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(device)
        image_processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
        image_encoder = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl").vision_model.to(device)

    elif mode == "text":
        clip_model, _ = clip.load("ViT-L/14@336px", device=device)
    # Initialize models
    config = load_config(f"configs/{mode}/{model_class}.yaml")
    model, _ = load_model_checkpoint(get_model(config).to(device), config['training']['save_path'])
    objects = get_object_class_caption() if model_class == "cbm-encoder" else None
    
    if custom:
        if img_dirs:
        # Process local directories
            for read_dir in img_dirs:
                real_dir = os.path.join("my_dataset/image", read_dir, "real")
                fake_dir = os.path.join("my_dataset/image", read_dir, "fake")

                batch = []
                predictions = []

                if os.path.exists(real_dir):
                    for image_name in tqdm(sorted(os.listdir(real_dir)), desc=f"Processing {real_dir} with {model_class}"):
                        image_path = os.path.join(real_dir, image_name)
                        image = Image.open(image_path).convert("RGB")
                        if model_class == "cbm-encoder":
                            predictions.append(process_cbm_encoder(model, image, objects, object_processor, object_detector, image_processor, image_encoder, device))
                        else:
                            image_tensor = torch.Tensor(image_processor(image).data['pixel_values'][0])
                            batch.append(image_tensor)
                            if len(batch) == batch_size:
                                predictions.append(process_img_linear(model, batch, image_encoder, device))
                                batch = []
                    if batch:
                        predictions.append(process_img_linear(model, batch, image_encoder, device))
                    save_predictions(torch.cat(predictions), output_dir, mode, model_class, read_dir, "real")
                    print(f"Predictions for real images in {read_dir} saved.")

                if os.path.exists(fake_dir):
                    for image_name in tqdm(sorted(os.listdir(fake_dir)), desc=f"Processing {real_dir} with {model_class}"):
                        image_path = os.path.join(fake_dir, image_name)
                        image = Image.open(image_path).convert("RGB")
                        if model_class == "cbm-encoder":
                            predictions.append(process_cbm_encoder(model, image, objects, object_processor, object_detector, image_processor, image_encoder, device))
                        else:
                            image_tensor = torch.Tensor(image_processor(image).data['pixel_values'][0])
                            batch.append(image_tensor)
                            if len(batch) == batch_size:
                                predictions.append(process_img_linear(model, batch, image_encoder, device))
                                batch = []
                    if batch:
                        predictions.append(process_img_linear(model, batch, image_encoder, device))
                    save_predictions(torch.cat(predictions), output_dir, mode, model_class, read_dir, "fake")
                    print(f"Predictions for fake images in {read_dir} saved.")
    else:
        # Process Hugging Face dataset
        dataset_name = "anson-huang/mirage-news"
        available_splits = load_dataset(dataset_name).keys()
        if test_only:
            available_splits = sorted(list(available_splits))[:5]
        for split in available_splits:
            dataset = load_dataset(dataset_name, split=split)
            for label in ["real", "fake"]:
                batch = []
                predictions = []
                filtered_dataset = [item for item in dataset if item["label"] == (0 if label == "real" else 1)]
                
                if mode == 'image':
                    for item in tqdm(filtered_dataset, desc=f"Processing {split}/{label} with {model_class}"):
                        image = item["image"].convert("RGB")
                        if model_class == "cbm-encoder":
                            predictions.append(process_cbm_encoder(model, image, objects, object_processor, object_detector, image_processor, image_encoder, device))
                        else:
                            image_tensor = torch.Tensor(image_processor(image).data['pixel_values'][0])
                            batch.append(image_tensor)
                            if len(batch) == batch_size:
                                predictions.append(process_img_linear(model, batch, image_encoder, device))
                                batch = []
                    if batch:
                        predictions.append(process_img_linear(model, batch, image_encoder, device))
                elif mode == 'text':
                    for item in tqdm(filtered_dataset, desc=f"Processing {split}/{label} with {model_class}"):
                        batch.append(item["text"])
                        if len(batch) == batch_size:
                            text_encoding = preprocess_texts(batch, clip_model, device)
                            predictions.append(process_txt_linear(model, text_encoding, device))
                            batch = []
                    if batch:
                        text_encoding = preprocess_texts(batch, clip_model, device)
                        predictions.append(process_txt_linear(model, text_encoding, device))
                save_predictions(torch.cat(predictions), output_dir, mode, model_class, split, label)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process local directories or Hugging Face datasets for encoding predictions")
    parser.add_argument("--mode", required=True, choices=["image", "text"], help="Specify 'image' or 'text'")
    parser.add_argument("--model_class", choices=["linear", "cbm-encoder", "merged", "custom"], help="Specify model class")
    parser.add_argument("--custom", action="store_true", help="Use local directories instead of Hugging Face dataset")
    parser.add_argument("--img_dirs", nargs="+", help="List of directories to read images from (if --custom is set)")
    parser.add_argument("--text_dirs", nargs="+", help="List of directories to read captions from (if --custom is set)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for processing images")
    parser.add_argument("--test_only", action="store_true", help="Encode only the test sets from the Hugging Face dataset")

    args = parser.parse_args()
    main(mode=args.mode, model_class=args.model_class, custom=args.custom, img_dirs=args.img_dirs, text_dirs=args.text_dirs, batch_size=args.batch_size, test_only=args.test_only)