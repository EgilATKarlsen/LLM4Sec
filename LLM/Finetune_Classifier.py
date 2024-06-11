from datasets import load_dataset, Features
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
import evaluate
import numpy as np

def finetune_model(dataset_name, model_name, data_files, output_dir_suffix="Anomaly"):
    # Load dataset
    dataset = load_dataset(dataset_name, use_auth_token=True, data_files=data_files)
    
    # Load and optimize model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def preprocess_function(examples):
        return tokenizer(examples["log"], truncation=True)
    
    tokenized_logs = dataset.map(preprocess_function, batched=True)
    tokenized_logs['train'] = tokenized_logs['train'].class_encode_column("label")
    
    id2label = {0: "Anomalous", 1: "Normal"}
    label2id = {"Anomalous": 0, "Normal": 1}
    
    tokenized_logs = tokenized_logs['train'].train_test_split(test_size=0.25, stratify_by_column="label")
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    accuracy = evaluate.load("accuracy")
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, id2label=id2label, label2id=label2id
    )
    
    training_args = TrainingArguments(
        output_dir=f"{model_name}_{output_dir_suffix}",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=True,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_logs["train"],
        eval_dataset=tokenized_logs["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()

# Example usage:
# finetune_model("EgilKarlsen/CSIC", "distilroberta-base", {"train": "csic.csv"})
