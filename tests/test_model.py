import torch
def compute_metrics(eval_pred):
    """
    Compute ROUGE metrics for model evaluation based on predictions and labels.

    Args:
        eval_pred: A tuple containing predictions and labels, each as a list of lists of integers.

    Returns:
        A dictionary mapping ROUGE metric names to their values multiplied by 100.
    """
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels)
    return {key: value * 100 for key, value in result.items()}
def evaluate_in_chunks(dataset, batch_size=1):
    """
    Evaluate a model on a dataset in chunks to manage memory usage efficiently.

    Args:
        dataset: A Hugging Face Dataset containing preprocessed input data and labels.
        batch_size: Number of examples to process per chunk (default: 1).

    Returns:
        A tuple of (predictions, labels), each as a list of lists of integers.
    """
    all_preds = []
    all_labels = []
    model.eval()

    for i in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset[i:i+batch_size]
        inputs = {
            k: torch.tensor(v).to("cuda") for k, v in batch.items() if k in ["input_ids", "attention_mask"]
        }
        labels = torch.tensor(batch["labels"]).to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=128,
                num_beams=4,
                early_stopping=True
            )
        all_preds.extend(outputs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        torch.cuda.empty_cache()

    return all_preds, all_labels

torch.cuda.empty_cache()
rouge = evaluate.load("rouge")

all_preds, all_labels = evaluate_in_chunks(tokenized_dataset["test"], batch_size=1)
metrics = compute_metrics((all_preds, all_labels))
print("ROUGEDZ Scores on Test Set:", metrics)
