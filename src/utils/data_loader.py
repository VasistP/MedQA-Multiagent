# src/utils/data_loader.py
import json
import random
from typing import List, Dict


def load_medqa_data(file_path: str, num_samples: int = None,
                    random_seed: int = 42) -> List[Dict]:
    """Load MedQA data from jsonl file"""
    random.seed(random_seed)

    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            data.append(item)

    # Randomly sample if requested
    if num_samples and num_samples < len(data):
        data = random.sample(data, num_samples)

    return data


def format_medqa_question(item: Dict) -> str:
    """Format MedQA item into a question string"""
    question = item['question']
    options = item['options']

    # Format options
    option_text = []
    for key in sorted(options.keys()):
        option_text.append(f"{key}) {options[key]}")

    formatted = f"{question}\n\nOptions:\n" + "\n".join(option_text)
    return formatted
