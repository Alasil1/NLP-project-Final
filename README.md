Dialogue Summarization with BART

Prerequisites

Python 3.11
CUDA-enabled GPU (optional, for faster training)
Access to the SAMSum dataset (available on Hugging Face or Kaggle)

Setup
1. Clone the Repository
git clone <https://github.com/Alasil1/NLP-project-Final>
cd dialogue_summarization

2. Create a Virtual Environment
Use venv or conda to create an isolated environment:
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install Dependencies
Install the required packages listed in requirements.txt:
pip install -r requirements.txt

Example requirements.txt:
torch>=2.0.0
transformers>=4.35.0
evaluate>=0.4.0
flask>=2.3.0
tqdm>=4.66.0

4. Prepare the Dataset

Download the SAMSum dataset from Hugging Face (samsum dataset) or Kaggle (e.g., dataset IDs 1246668 and 6004344).
Place the raw dataset files in data/raw/.
Preprocess and tokenize the dataset (refer to the original notebook or create a separate preprocessing script). The tokenized dataset should be saved in data/processed/tokenized_samsum/.


Usage
1. Train the Model
Run the main script to train the BART model:
python src/train_bart_samsum.py


The script will:
Fine-tune the facebook/bart-base model on the SAMSum dataset.
Save the trained model to models/bart-samsum/model_v1/ with metadata.
Evaluate the model on the test set and print ROUGE scores.
Create a zip archive of the model in models/bart-samsum.zip.


2. Evaluate the Model
The evaluation is included in the main script execution. ROUGE scores (rouge1, rouge2, rougeL, rougeLsum) are printed after evaluation on the test set.
3. Use the Summarization API
The script starts a Flask API on http://0.0.0.0:5000 . To summarize a dialogue:

Send a POST request to the /summarize endpoint 
for example: 

$dialogue = @"
rony: Hey, do you wanna go out tomorrow?
alasil: Lemme check
alasil: Sorry, i won't be able to go out as i will be with my family 
rony: what about Thursday 
alasil : yes Thursday is ok 
rony : ok it is Thursday then 
alasil: bye
rony : bye
"@

$body = @{
    dialogue = $dialogue
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:5000/summarize" -Method POST -ContentType "application/json" -Body $body


Response example:{"summary": "Alice and Bob plan to meet at 6 PM at the cafe."}


Notes

Ensure the tokenized dataset exists in data/processed/tokenized_samsum/.
Training uses mixed precision (fp16=True) and requires a GPU for optimal performance.
The API runs in a separate thread; stop the script with Ctrl+C to terminate.

