from transformers import BartForConditionalGeneration,DataCollatorForSeq2Seq

model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)