from lmdeploy import pipeline, TurbomindEngineConfig, ChatTemplateConfig
from lmdeploy.vl import load_image
import os
import pandas as pd
import json
from tqdm import tqdm
import pickle

# Get the absolute path to the workspace root
workspace_root = "/mnt/scratch1/beyza"

model = 'OpenGVLab/InternVL3-8B'
dataset_path = os.path.join(workspace_root, "muse_dataset")
images_folder = os.path.join(dataset_path, "images")

pipe = pipeline(model, backend_config=TurbomindEngineConfig(session_len=16384, tp=1), 
               chat_template_config=ChatTemplateConfig(model_name='internvl2_5'))

def process_split(split_name, file_name):
    file_path = os.path.join(dataset_path, file_name)
    print(f"\nLoading {split_name} data from {file_path} ...")
    data = pd.read_csv(file_path, sep='\t', header=None, names=['tweet_id', 'text', 'label'])
    data['tweet_id'] = data['tweet_id'].astype(str)
    data['label'] = data['label'].astype(str)
    print(f"Loaded {len(data)} {split_name} samples")
    print(f"First few tweet IDs: {data['tweet_id'].head().tolist()}")
    print(f"Images folder contents: {os.listdir(images_folder)[:5]}")

    results = []
    skipped = 0
    processed = 0

    for idx, row in tqdm(data.iterrows(), desc=f"Processing {split_name} images", total=len(data)):
        try:
            tweet_id = row['tweet_id']
            label = row['label']
            # Find corresponding image
            image_path = None
            for img in os.listdir(images_folder):
                if tweet_id in img:
                    image_path = os.path.join(images_folder, img)
                    break
            if image_path is None:
                skipped += 1
                continue
            try:
                image = load_image(image_path)
            except Exception as e:
                skipped += 1
                continue
            try:
                response = pipe(('describe this image in detail, but keep the description under 250 tokens. Focus on the visual elements, composition, and any notable features.', image))
                description = response.text
            except Exception as e:
                skipped += 1
                continue
            results.append({
                "image_id": tweet_id,
                "text": row['text'],
                "label": label,
                "image_description": description,
                "image_location": image_path
            })
            processed += 1
        except Exception as e:
            skipped += 1
            continue
    print(f"\nProcessing complete for {split_name}:")
    print(f"Total images in {split_name} set: {len(data)}")
    print(f"Successfully processed: {processed}")
    print(f"Skipped: {skipped}")
    results_df = pd.DataFrame(results)
    print(f"\nResults DataFrame shape: {results_df.shape}")
    print("First few rows of results:")
    print(results_df.head())
    # Try multiple saving methods
    output_file = f"muse_image_descriptions_{split_name}_internvl.csv"
    backup_file = f"muse_image_descriptions_{split_name}_internvl_backup.json"
    try:
        results_df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"\nResults saved to {output_file}")
        results_df.to_json(backup_file, orient='records', lines=True)
        print(f"Backup saved to {backup_file}")
        pickle_file = f"muse_image_descriptions_{split_name}_internvl.pkl"
        results_df.to_pickle(pickle_file)
        print(f"Binary backup saved to {pickle_file}")
        if os.path.exists(output_file):
            saved_df = pd.read_csv(output_file)
            print(f"\nVerification - CSV file shape: {saved_df.shape}")
            print("First few rows of CSV file:")
            print(saved_df.head())
        else:
            print(f"\nError: CSV file {output_file} was not created!")
        if os.path.exists(backup_file):
            json_df = pd.read_json(backup_file, lines=True)
            print(f"\nVerification - JSON file shape: {json_df.shape}")
        else:
            print(f"\nError: JSON backup file {backup_file} was not created!")
    except Exception as e:
        print(f"\nError saving results: {str(e)}")
        print("Current working directory:", os.getcwd())
        print("Available disk space:", os.statvfs('.').f_bavail * os.statvfs('.').f_frsize / (1024*1024), "MB")
    print("\nImage Location Distribution:")
    location_counts = results_df['image_location'].apply(lambda x: os.path.dirname(x)).value_counts()
    for folder, count in location_counts.items():
        print(f"{folder}: {count} images")

# Process train and val splits
dataset_splits = {
    'train': 'train_muse.csv',
    'val': 'val_muse.csv'
}
for split_name, file_name in dataset_splits.items():
    process_split(split_name, file_name)
