import argparse
from LLM.Finetune_Classifier import finetune_model

def main():
    parser = argparse.ArgumentParser(description="Fine-tune a Hugging Face model for sequence classification.")
    
    parser.add_argument("--dataset_name", type=str, required=True, help="The name of the dataset to use.")
    parser.add_argument("--model_name", type=str, required=True, help="The name of the pre-trained model to fine-tune.")
    parser.add_argument("--data_files", type=str, required=True, help="The path to the data files.")
    parser.add_argument("--output_dir_suffix", type=str, default="Anomaly", help="Suffix for the output directory.")
    
    args = parser.parse_args()
    
    data_files = {"train": args.data_files}
    
    finetune_model(args.dataset_name, args.model_name, data_files, args.output_dir_suffix)

if __name__ == "__main__":
    main()
