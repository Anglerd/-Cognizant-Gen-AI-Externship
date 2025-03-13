# Import libraries
try:
    import numpy as np
    from datasets import load_dataset
    import evaluate
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
        EarlyStoppingCallback
    )
except ImportError:
    import sys
    required_libs = {
        'datasets',
        'transformers',
        'numpy',
        'evaluate',
        'accelerate',
        'tf-keras'
    }

    missing = set()
    for lib in required_libs:
        try:
            __import__(lib)
        except ImportError:
            missing.add(lib)
    print("Missing required libraries:", ", ".join(missing))
    response = input("Would you like to install them? (y/n): ").lower()
    if response == 'y':
        import subprocess
        try:
            for lib in missing:
                subprocess.check_call([sys.executable, "-m", "pip", "install", lib])
            print("Libraries installed. Please restart the program.")
        except:
            print("Error while installing required libraries.")
            print("For windows the most likely issue is not having proper privileges. Please try running a new window as administrator and running the following command")
            print("python -m pip install", " ".join(missing))
        sys.exit()
    else:
        print("Exiting program. Please install the required libraries with the following command")
        print("python -m pip install", " ".join(missing))
        sys.exit()

# Part 1: Fine-Tuning BERT
# Load and prepare dataset
dataset = load_dataset('imdb')
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=256)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

# Initialize model
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Set up training arguments with debugging considerations
training_args = TrainingArguments(
    output_dir="./bert_imdb",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

# Set up evaluation metrics
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy.compute(predictions=predictions, references=labels)["accuracy"],
        "f1": f1.compute(predictions=predictions, references=labels)["f1"]
    }

# Initialize Trainer with early stopping for debugging overfitting
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# Train and save model
train_result = trainer.train()
trainer.save_model("./bert_imdb_final")

# Part 3: Evaluation
# Evaluate on full test set
full_test_dataset = tokenized_datasets["test"].select(range(1000))
metrics = trainer.evaluate(full_test_dataset)
print(f"Final evaluation metrics: {metrics}")

# Part 4: Creative Application - Inference Example
def predict_sentiment(text):
    inputs = tokenizer(text, padding=True, truncation=True, max_length=256, return_tensors="pt")
    outputs = model(**inputs)
    probs = outputs.logits.softmax(dim=-1).detach().numpy()
    return {"label": "Positive" if np.argmax(probs) == 1 else "Negative", "confidence": np.max(probs)}

# Test prediction
sample_review = "This movie was an incredible journey from start to finish. The acting was superb and the cinematography breathtaking!"
print(predict_sentiment(sample_review))
