from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="/kaggle/working/results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=True,
    logging_dir="/kaggle/working/logs",
    logging_steps=100,
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="none",
    run_name="bart-samsum-training",

)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["val"],
    data_collator=data_collator,
)

trainer.train()