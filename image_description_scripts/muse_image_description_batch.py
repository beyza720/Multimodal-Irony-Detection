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

if hasattr(processor, 'tokenizer'):
    processor.tokenizer.padding_side = 'left'
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

# Define paths
dataset_path = "/mnt/scratch1/beyza/muse_dataset"
image_folder = os.path.join(dataset_path, "images")
train_file = os.path.join(dataset_path, "train_muse_cleaned.csv")

def find_image_in_folder(image_id, folder):
    image_path = os.path.join(folder, image_id)
    if os.path.exists(image_path):
        return image_path
    return None

def process_batch(batch_data, batch_size=16):
    images = []
    valid_rows = []
    
    for row_data in batch_data:
        image_id = str(row_data["image_id"]) + ".jpg"
        image_path = find_image_in_folder(image_id, image_folder)
        
        if not image_path:
            continue
            
        try:
            image = Image.open(image_path).convert("RGB")
            images.append(image)
            valid_rows.append(row_data)
        except Exception as e:
            print(f"Error loading image {image_path}: {str(e)}")
            continue
    
    if not images:
        return []
    
    batch_messages = []
    for i in range(len(images)):
        prompt = (
            "Please describe this image in detail. Focus on the visual elements, "
            "composition, and any notable features. Keep the description under 250 tokens."
        )
        
        messages = [
            {"role": "system", "content": [{"type": "text", "text": prompt}]},
            {"role": "user", "content": [
                {"type": "image", "image": images[i]},
                {"type": "text", "text": "Please describe this image."}
            ]},
        ]
        batch_messages.append(messages)
    
    text_inputs = [processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) 
                   for msgs in batch_messages]
    
    inputs = processor(
        text=text_inputs,
        images=images,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=250,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            do_sample=False,
            temperature=None,  
        )
        generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        cleaned_texts = []
        for text in generated_texts:
            if "assistant" in text.lower():
                text = text.split("assistant")[-1].strip()
            if "user" in text.lower():
                text = text.split("user")[-1].strip()
            if "system" in text.lower():
                text = text.split("system")[-1].strip()
            cleaned_texts.append(text.strip())

    del inputs
    del generated_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    results = []
    for i, row_data in enumerate(valid_rows):
        image_id = str(row_data["image_id"]) + ".jpg"
        image_path = find_image_in_folder(image_id, image_folder)
        
        results.append({
            "image_id": image_id,
            "text": str(row_data["text"]).strip(),
            "label": int(row_data["label"]),
            "image_description": cleaned_texts[i],
            "image_location": image_path
        })
    
    return results

print("Loading train data...")
train_df = pd.read_csv(train_file, sep='\t', header=None, names=['image_id', 'text', 'label'])

valid_image_ids = set(str(img_id) + ".jpg" for img_id in train_df["image_id"])
print(f"Found {len(valid_image_ids)} images in train.csv")

results = []
skipped = 0
processed = 0
batch_size = 16  # Process 16 images at once - optimal balance

data_list = []
for _, row in train_df.iterrows():
    image_id = str(row["image_id"]) + ".jpg"
    if image_id in valid_image_ids:
        data_list.append({
            "image_id": row["image_id"],
            "text": row["text"],
            "label": row["label"]
        })

print(f"Processing {len(data_list)} valid images in batches of {batch_size}")

total_batches = (len(data_list) + batch_size - 1) // batch_size
for i in tqdm(range(0, len(data_list), batch_size), total=total_batches, desc="Processing batches"):
    try:
        batch_data = data_list[i:i+batch_size]
        batch_results = process_batch(batch_data, batch_size)
        
        results.extend(batch_results)
        processed += len(batch_results)
        skipped += len(batch_data) - len(batch_results)
        
        if (i // batch_size + 1) % 10 == 0:
            temp_df = pd.DataFrame(results)
            temp_df.to_csv("muse_image_descriptions_train_temp.csv", index=False)
            print(f"Saved intermediate results: {len(results)} images processed")
        
    except Exception as e:
        print(f"Error processing batch starting at index {i}: {str(e)}")
        skipped += len(batch_data)
        continue

print(f"\nProcessing complete:")
print(f"Total valid images: {len(data_list)}")
print(f"Successfully processed: {processed}")
print(f"Skipped: {skipped}")

results_df = pd.DataFrame(results)
results_df.to_csv("muse_image_descriptions_train_batch.csv", index=False)


print(f"\nFinal results saved to: muse_image_descriptions_train_batch.csv") 