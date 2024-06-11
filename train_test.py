import argparse
from datasets import load_dataset
import numpy as np
from huggingface_hub import login
import json
import importlib

def get_model(model_name, **kwargs):
    module = importlib.import_module(f'models.{model_name}')
    model_class = getattr(module, model_name)
    return model_class(**kwargs)

def main(data, model_name, model_params, results):
    dataset = load_dataset(f'EgilKarlsen/{data}', use_auth_token=True)

    train_dataset = dataset['train']
    test_dataset = dataset['test']

    # Features excluding 'label'
    features_excluding_label = [feature for feature in train_dataset.features if feature != 'label']

    # Convert to pandas DataFrames
    train_df = train_dataset.to_pandas()
    test_df = test_dataset.to_pandas()

    # Extract features and labels
    X_train = train_df[features_excluding_label].values
    X_test = test_df[features_excluding_label].values
    y_train = train_df['label'].values
    y_test = test_df['label'].values

    # Build model
    model = get_model(model_name, **model_params)
    model.fit(X_train, y_train)

    # Evaluate
    train_pred, train_proba = model.predict(X_train)
    eval = model.evaluate(y_train, train_pred, train_proba, 'train')
    test_pred, test_proba = model.predict(X_test)
    eval = model.evaluate(y_test, test_pred, test_proba, 'test')
    print(eval['test']['F-Measure'])
    results[f'{data}'] = eval
    model.save_instance(f'{data}_IND.pkl')
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate models on datasets.")
    parser.add_argument('--datasets', nargs='+', required=True, help='List of datasets to use.')
    parser.add_argument('--model_name', required=True, help='Name of the model to use (e.g., GBoost, KMeans, NBayes).')
    parser.add_argument('--model_params', type=json.loads, default='{}', help='JSON string of model parameters.')

    args = parser.parse_args()

    login(token="hf_token_here", write_permission=True)

    results = {}
    for data in args.datasets:
        results = main(data, args.model_name, args.model_params, results)
