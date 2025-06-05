import pandas as pd
import argparse
from pathlib import Path

def combine_texts(csv_path, output_path):
    df = pd.read_csv(csv_path)
    
    df['combined_text'] = df['text'] + ' ' + df['image_description']
    
    df.to_csv(output_path, index=False)
    print(f"Combined texts and saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input CSV file path")
    parser.add_argument("--output", type=str, required=True, help="Output CSV file path")
    args = parser.parse_args()

    combine_texts(args.input, args.output) 