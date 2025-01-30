import argparse
import torch
import os
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from transformers import Owlv2Processor, Owlv2ForObjectDetection, Blip2Processor, Blip2ForConditionalGeneration
from datasets import load_dataset
from data import get_object_class_caption, get_preprocessed_image

def save_crops(image, boxes, phrases, image_name, split, image_type, output_dir):
    """Save cropped objects from the image into a structured directory."""
    image_width, image_height = image.size  # Get image dimensions
    
    for i, box in enumerate(boxes):
        # Convert box to integer and ensure coordinates are within the image dimensions
        x_min, y_min, x_max, y_max = map(int, box.tolist())
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(image_width, x_max)
        y_max = min(image_height, y_max)

        # Skip very small boxes
        if (x_max - x_min) < 64 or (y_max - y_min) < 64:
            continue

        # Crop the image based on the adjusted box
        crop_img = image.crop((x_min, y_min, x_max, y_max))
        
        # Create directory structure for saving
        class_dir = os.path.join(output_dir, phrases[i], split, image_type)
        os.makedirs(class_dir, exist_ok=True)

        # Save the crop
        crop_path = os.path.join(class_dir, f"{image_name}_{image_type}_{str(i).zfill(2)}.jpg")
        crop_img.save(crop_path)

def annotate_image(image, boxes, phrases, output_path="annotated_image.jpg"):
    """Draw bounding boxes and labels on the image and save it."""
    draw = ImageDraw.Draw(image)
    
    # Define font for the labels
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()
    
    # Draw each bounding box with label
    for i, box in enumerate(boxes):
        x_min, y_min, x_max, y_max = map(int, box.tolist())
        label = phrases[i]
        
        # Draw bounding box
        draw.rectangle([(x_min, y_min), (x_max, y_max)], outline="red", width=2)
        
        # Draw label
        text_size = draw.textsize(label, font=font)
        draw.rectangle([(x_min, y_min - text_size[1]), (x_min + text_size[0], y_min)], fill="red")
        draw.text((x_min, y_min - text_size[1]), label, fill="white", font=font)
    
    image.save(output_path)
    print(f"Annotated image saved to {output_path}")

def encode_and_save_crops(model, processor, device, output_dir, encodings_dir, batch_size=32):
    """Encode all saved crops in output_dir and save the encodings to encodings_dir with the same structure."""
    for root, _, files in os.walk(output_dir):
        if files:
            encodings = []
            batch = []
            for crop_filename in tqdm(sorted(files), desc=f"Encoding crops in {root}"):
                crop_path = os.path.join(root, crop_filename)
                image = processor(Image.open(crop_path).convert("RGB")).data['pixel_values'][0]
                batch.append(torch.Tensor(image))
                
                if len(batch) == batch_size:
                    encodings.append(process_batch(batch, model, device))
                    batch = []

            if batch:
                encodings.append(process_batch(batch, model, device))

            # Determine the output path for encoding, mirroring the structure of output_dir within encodings_dir
            relative_path = os.path.relpath(root, output_dir)
            encoding_output_path = os.path.join(encodings_dir, f"{relative_path}.pt")
            os.makedirs(os.path.dirname(encoding_output_path), exist_ok=True)
            torch.save(torch.cat(encodings), encoding_output_path)
            print(f"Saved encodings to {encoding_output_path}")

def process_batch(images, model, device):
    """Process a batch of images through the model and extract embeddings."""
    images_tensor = torch.stack(images).to(device)
    with torch.no_grad():
        return model(images_tensor).pooler_output

# Main processing function

def process_dataset_or_directory(custom=False, read_dirs=None, output_dir="data/crops", encodings_dir="encodings/crops", batch_size=128):
    device = "cuda"
    
    # Load models and processors
    object_processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    object_model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(device)
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl").vision_model.to(device)
    objects = get_object_class_caption()

    # Process custom directories
    if custom and read_dirs:
        for read_dir in read_dirs:
            for image_type in ["real", "fake"]:
                dir_path = os.path.join(read_dir, image_type)
                if os.path.exists(dir_path):
                    for idx, image_name in enumerate(tqdm(sorted(os.listdir(dir_path)), desc=f"Processing {image_type} in {read_dir}")):
                        image = Image.open(os.path.join(dir_path, image_name)).convert("RGB")
                        
                        # Object detection
                        inputs = object_processor(text=objects, images=image, return_tensors="pt").to(device)
                        unnormalized_image = get_preprocessed_image(inputs.pixel_values.cpu())
                        outputs = object_model(**inputs)
                        target_sizes = torch.Tensor([unnormalized_image.size[::-1]])
                        results = object_processor.post_process_object_detection(outputs=outputs, threshold=0.2, target_sizes=target_sizes)[0]

                        # Save crops
                        boxes, labels = results["boxes"].cpu(), results["labels"]
                        phrases = [objects[i].split("of ", 1)[1].replace(' ', '_') for i in labels]
                        save_crops(unnormalized_image, boxes, phrases, image_name, read_dir, image_type, output_dir)
                        

    # Process HF dataset
    else:
        dataset_name = "anson-huang/mirage-news"
        available_splits = load_dataset(dataset_name).keys()
        for split in available_splits:
            dataset = load_dataset(dataset_name, split=split)
            for idx, item in enumerate(tqdm(dataset, desc=f"Processing split {split}")):
                image = item["image"].convert("RGB")
                label = item["label"]
                image_type = "real" if label == 0 else "fake"
                
                # Object detection
                inputs = object_processor(text=objects, images=image, return_tensors="pt").to(device)
                unnormalized_image = get_preprocessed_image(inputs.pixel_values.cpu())
                outputs = object_model(**inputs)
                target_sizes = torch.Tensor([unnormalized_image.size[::-1]])
                results = object_processor.post_process_object_detection(outputs=outputs, threshold=0.2, target_sizes=target_sizes)[0]

                # Save crops
                boxes, labels = results["boxes"].cpu(), results["labels"]
                phrases = [objects[i].split("of ", 1)[1].replace(' ', '_') for i in labels]
                save_crops(unnormalized_image, boxes, phrases, f"{split}_{idx}", split, image_type, output_dir)
                
    # Encode all saved crops after saving is complete
    encode_and_save_crops(model, processor, device, output_dir, encodings_dir, batch_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process HF dataset or custom image directory for object detection and encoding")
    parser.add_argument("--custom", action="store_true", help="Use local directories instead of Hugging Face dataset")
    parser.add_argument("--read_dirs", nargs="+", help="List of directories to read images from (if --custom is set)")
    parser.add_argument("--output_dir", type=str, default="data/crops", help="Directory to save the cropped images")
    parser.add_argument("--encodings_dir", type=str, default="encodings/crops", help="Directory to save the image encodings")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing images")

    args = parser.parse_args()
    process_dataset_or_directory(
        custom=args.custom,
        read_dirs=args.read_dirs,
        output_dir=args.output_dir,
        encodings_dir=args.encodings_dir,
        batch_size=args.batch_size
    )
