import os
import json
from datasets import load_dataset

def prepare_financial_data():
    print("Loading finance-alpaca dataset from Hugging Face...")

    # Load the dataset
    dataset = load_dataset("gbharti/finance-alpaca", split="train")
    subset_dataset = dataset.select(range(5000))

    output_dir = "../data"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "financial_instructions.jsonl")

    print(f"Formatting and saving {len(subset_dataset)} records to {output_file}...")

    with open(output_file, "w", encoding="utf-8") as f:
        for item in subset_dataset:
            #create instruction format
            formatted_item = {
                "instruction": item["instruction"],
                "input": item["input"] if item["input"] else "",
                "output": item["output"]
            }
            f.write(json.dumps(formatted_item) + "\n")
        
    print("Data preparation complete.")

if __name__ == "__main__":
    prepare_financial_data()

