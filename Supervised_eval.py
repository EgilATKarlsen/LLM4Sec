import argparse
import json
import re
import matplotlib.pyplot as plt
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from eval.Supervised import evaluate  # Import the evaluate function

def replace_substring(original_string, old_substring, new_substring):
    replaced_string = original_string.replace(old_substring, new_substring)
    return replaced_string

def extract_dataset(s):
    match = re.search(r'_(.*?)-', s)
    if not match:
        return None
    substr = match.group(1)

    mapping = {
        'AA': 'EgilKarlsen/AA',
        'PKDD': 'EgilKarlsen/PKDD',
        'CSIC': 'EgilKarlsen/CSIC',
        'Spirit': 'EgilKarlsen/Spirit_50K',
        'BGL': 'EgilKarlsen/BGL_50K',
        'Thunderbird': 'EgilKarlsen/Thunderbird_50K'
    }

    return mapping.get(substr)

def convert_specific_keys_to_list(input_data):
    if isinstance(input_data, dict):
        for key, value in input_data.items():
            if key in ["labels", "probabilities"] and isinstance(value, (np.ndarray, np.generic)):
                input_data[key] = value.tolist()
            else:
                convert_specific_keys_to_list(value)
    return input_data

def convert_keys_to_string(data_dict):
    for key, model_data in data_dict.items():
        for metric in ['class_precision', 'class_recall', 'class_f1']:
            if metric in model_data:
                model_data[metric] = {str(k): v for k, v in model_data[metric].items()}
    return data_dict

def main():
    parser = argparse.ArgumentParser(description="Evaluate a model on a specific dataset")
    parser.add_argument('--dataset', type=str, required=True, help='Dataset to evaluate on')
    parser.add_argument('--model', type=str, required=True, help='Model to evaluate')
    args = parser.parse_args()

    dataset_name = args.dataset
    model_name = args.model

    dataset = load_dataset(extract_dataset(model_name), use_auth_token=True)
    test_dataset = dataset['test']
    test_dataset = test_dataset.class_encode_column("label")

    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if "GPT" in model_name:
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token

    data = evaluate(model, tokenizer, test_dataset, model_name.split('/', 1)[1])

    plt.savefig(dataset_name + '.png')

    test = convert_specific_keys_to_list(data)
    test = convert_keys_to_string(test)

    with open(dataset_name + ".json", 'w') as f:
        json.dump(test, f)

if __name__ == "__main__":
    main()
