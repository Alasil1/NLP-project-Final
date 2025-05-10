**Dialogue Summarization with BART
**

dialogue_summarization/
├── data/
│   ├── raw/                  # Raw SAMSum dataset
│   └── processed/            # Processed and tokenized datasets
│       └── tokenized_samsum/ # Tokenized SAMSum dataset
├── models/
│   └── bart-samsum/          # Trained model and logs
│       ├── model_v1/         # Versioned model directory
│       ├── model_v1_metadata.txt # Model metadata
│       └── logs/             # Training logs
├── src/
│   └── train_bart_samsum.py  # Main script for training, evaluation, and API
├── README.md                 # Project documentation
└── requirements.txt          # Project dependencies

Prerequisites

Python 3.11
CUDA-enabled GPU (optional, for faster training)
A valid ngrok authentication token (for API exposure)
Access to the SAMSum dataset (available on Hugging Face or Kaggle)

Setup
1. Clone the Repository
git clone <repository-url>
cd dialogue_summarization

2. Create a Virtual Environment
Use venv or conda to create an isolated environment:
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# OR using conda
conda create -n dialogue_summarization python=3.11
conda activate dialogue_summarization

3. Install Dependencies
Install the required packages listed in requirements.txt:
pip install -r requirements.txt

Example requirements.txt:
torch>=2.0.0
transformers>=4.35.0
evaluate>=0.4.0
flask>=2.3.0
pyngrok>=7.0.0
tqdm>=4.66.0

4. Prepare the Dataset

Download the SAMSum dataset from Hugging Face (samsum dataset) or Kaggle (e.g., dataset IDs 1246668 and 6004344).
Place the raw dataset files in data/raw/.
Preprocess and tokenize the dataset (refer to the original notebook or create a separate preprocessing script). The tokenized dataset should be saved in data/processed/tokenized_samsum/.

5. Configure ngrok

Obtain a valid ngrok authentication token from ngrok.com.
Update the NGROK_AUTH_TOKEN constant in src/train_bart_samsum.py with your token.

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
The script starts a Flask API on http://0.0.0.0:5000 and exposes it via ngrok. To summarize a dialogue:

Send a POST request to the /summarize endpoint with a JSON payload:curl -X POST -H "Content-Type: application/json" \
     -d '{"dialogue": "Alice: Hi, want to meet up later? Bob: Sure, how about 6 PM at the cafe?"}' \
     <ngrok-public-url>/summarize


Response example:{"summary": "Alice and Bob plan to meet at 6 PM at the cafe."}


The ngrok public URL is printed when the script runs.

Notes

Ensure the tokenized dataset exists in data/processed/tokenized_samsum/.
Training uses mixed precision (fp16=True) and requires a GPU for optimal performance.
The API runs in a separate thread; stop the script with Ctrl+C to terminate.

Contribution
We welcome contributions to improve this project. Please follow these guidelines:
1. Coding Standards

Adhere to PEP 8 and PEP 257 for style and docstrings.
Use type hints with the typing module.
Format code with black and lint with flake8:black src/
flake8 src/


Write meaningful comments for complex logic.
Follow the project structure (e.g., save models in models/, data in data/).

2. Submission Process

Fork the repository and create a feature branch:git checkout -b feature/your-feature-name


Commit changes with clear messages:git commit -m "Add feature: description of changes"


Push to your fork and submit a pull request to the main repository.
Include a description of changes and reference any related issues.

3. Testing

Test changes locally to ensure training, evaluation, 아직 and API functionality work as expected.
Add unit tests for new functions in src/ (optional but encouraged).

License
This project is licensed under the MIT License. See the LICENSE file for details.
Contact
For questions or issues, please open an issue on the repository or contact the project maintainer at your-email@example.com.
