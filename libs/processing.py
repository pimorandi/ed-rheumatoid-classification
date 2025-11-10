from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
import evaluate

def prepare_model(model_name="dbmdz/bert-base-italian-uncased", freeze_base=True):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    if freeze_base:
        for param in model.base_model.parameters():
            param.requires_grad = False
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    return model, tokenizer, data_collator

def preprocess_function(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True)


def compute_metrics(eval_pred):
    logit, labels = eval_pred
    predictions = [1 if x > 0.5 else 0 for x in logit]
    return evaluate.load("precision").compute(predictions=predictions, references=labels)