import pandas as pd
import os
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
import torch
from PIL import Image
from tqdm import tqdm
import gc

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

if torch.cuda.is_available():
    torch.cuda.empty_cache()

device_map = {"": 0} if torch.cuda.is_available() else "auto"
model_name = "Qwen/Qwen2.5-VL-7B-Instruct"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto", 
    quantization_config=quantization_config,
    offload_folder="offload",  
    trust_remote_code=True,
    max_memory={0: "32GB", "cpu": "16GB"}
)

model.gradient_checkpointing_enable()
processor = AutoProcessor.from_pretrained(model_name)

# Define paths
dataset_path = "/mnt/scratch1/beyza/mmsd_dataset"
image_folders = [
    os.path.join(dataset_path, f"extracted_part_{i}") for i in range(1, 7)
]
train_file = os.path.join(dataset_path, "train.json")

def find_image_in_folders(image_id, folders):
    """Search for an image across multiple folders and return the first match."""
    for folder in folders:
        image_path = os.path.join(folder, image_id)
        if os.path.exists(image_path):
            return image_path
    return None

# Read train data
print("Loading train data...")
train_df = pd.read_json(train_file)

# Get list of valid image IDs from train set
valid_image_ids = set(str(img_id) + ".jpg" for img_id in train_df["image_id"])
print(f"Found {len(valid_image_ids)} images in train.json")

results = []
skipped = 0
processed = 0

for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Processing train images"):
    try:
        image_id = str(row["image_id"]) + ".jpg"
        
        # Skip if image is not in train set
        if image_id not in valid_image_ids:
            print(f"Skipping {image_id} - not in train set")
            skipped += 1
            continue
            
        text = str(row["text"]).strip()
        true_label = int(row["label"])
        
        # Search for image in all folders
        image_path = find_image_in_folders(image_id, image_folders)
        if not image_path:
            print(f"Skipping {image_id} - not found in any folder")
            skipped += 1
            continue
        
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_path}: {str(e)}")
            skipped += 1
            continue

        prompt = (
            "Please describe this image in detail. Focus on the visual elements, "
            "composition, and any notable features. Keep the description under 250 tokens."
        )

        messages = [
            {"role": "system", "content": [{"type": "text", "text": prompt}]},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Please describe this image."}
            ]},
        ]

        text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = processor(
            text=[text_input],
            images=[image],
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=250,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
            )
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            
            # Clean up the generated text by removing system prompt and user/assistant markers
            if "assistant" in generated_text.lower():
                generated_text = generated_text.split("assistant")[-1].strip()
            if "user" in generated_text.lower():
                generated_text = generated_text.split("user")[-1].strip()
            if "system" in generated_text.lower():
                generated_text = generated_text.split("system")[-1].strip()
            generated_text = generated_text.strip()

        # Clear memory
        del inputs
        del generated_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        results.append({
            "image_id": image_id,
            "text": text,
            "label": true_label,
            "image_description": generated_text,
            "image_location": image_path
        })
        
        processed += 1

    except Exception as e:
        print(f"Error processing row: {str(e)}")
        skipped += 1
        continue

print(f"\nProcessing complete:")
print(f"Total images in train set: {len(train_df)}")
print(f"Successfully processed: {processed}")
print(f"Skipped: {skipped}")

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv("mmsd_image_descriptions_train.csv", index=False)

# Print distribution of where images were found
print("\nImage Location Distribution:")
location_counts = results_df['image_location'].apply(lambda x: x.split('/')[-2]).value_counts()
for folder, count in location_counts.items():
    print(f"{folder}: {count} images")
