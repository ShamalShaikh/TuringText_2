
import os
from transformers import AutoModelForSequenceClassification, Trainer, DataCollatorWithPadding, AutoTokenizer
from datasets import Dataset
from scipy.special import softmax
import numpy as np
import evaluate

train_path = "..\\..\\subtaskA\\dataset\\subtaskA_train_monolingual.jsonl"
model_name = "bert-base-uncased"
id2label = {0: "human", 1: "machine"}
label2id = {"human": 0, "machine": 1} 


def preprocess_function(examples, **fn_kwargs):
    return fn_kwargs['tokenizer'](examples["text"], truncation=True)

def compute_metrics(eval_pred):

    f1_metric = evaluate.load("f1")

    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    results = {}
    results.update(f1_metric.compute(predictions=predictions, references = labels, average="micro"))

    return results
# load tokenizer from saved model 
tokenizer = AutoTokenizer.from_pretrained(model_name)

# load best model
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=len(label2id), id2label=id2label, label2id=label2id
)

train_dataset = Dataset.from_pandas(train_path)

tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True,  fn_kwargs={'tokenizer': tokenizer})
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# create Trainer
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)


# get logits from predictions and evaluate results using classification report
predictions = trainer.predict(tokenized_train_dataset)
prob_pred = softmax(predictions.predictions, axis=-1)
preds = np.argmax(predictions.predictions, axis=-1)
metric = evaluate.load("bstrai/classification_report")
results = metric.compute(predictions=preds, references=predictions.label_ids)

# return dictionary of classification report
print(results, preds)