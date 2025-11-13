"""
Preprocess the dataset to parquet format
"""

import re
import os
import json
import numpy as np
import pandas as pd
import argparse

np.random.seed(2025)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='xxx')
    args = parser.parse_args()
    
    data_source = 'toolace'

    # Load dataset
    dataset = json.load(open(os.path.join(args.local_dir, "xxx.json"), "r"))
    # Shuffle dataset
    np.random.shuffle(dataset)

    # Split into train and test sets (2% test data)
    test_num = int(len(dataset) * 0.02)
    test_dataset = dataset[-test_num:]
    train_dataset = dataset[:-test_num]

    # Function to process each example
    def process_fn(example, idx, split):
        instruction = example["instruction"]
        input_text = example["input"]
        output = example["output"]

        data = {
            "data_source": data_source,
            "prompt": [
                {"role": "system", "content": instruction},
                {"role": "user", "content": input_text},
            ],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": output
            },
            "extra_info": {
                'split': split,
                "instruction": instruction,
                'index': idx,
                "input": input_text,
                "output": output,
            }
        }
        return data

    # Process dataset using list comprehension
    train_dataset = [process_fn(d, idx, 'train') for idx, d in enumerate(train_dataset)]
    test_dataset = [process_fn(d, idx, 'test') for idx, d in enumerate(test_dataset)]

    # Convert to Pandas DataFrame
    train_df = pd.DataFrame(train_dataset)
    test_df = pd.DataFrame(test_dataset)

    print("The sample of train data", len(train_df))
    print("The sample of test data", len(test_df))

    # Save as Parquet
    local_dir = args.local_dir
    os.makedirs(local_dir, exist_ok=True)

    train_df.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_df.to_parquet(os.path.join(local_dir, 'test.parquet'))

    print(f"Saved datasets to {local_dir}")