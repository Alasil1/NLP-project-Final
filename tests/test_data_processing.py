import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
def text_cleaner(text):
    """
    Clean the input text by removing HTML tags, non-word characters (except for specified punctuation),
    extra whitespace, and converting to lowercase.
    Args:
        text (str): The input text to be cleaned.

    Returns:
        str: The cleaned text.
    """
    text = BeautifulSoup(text, "lxml").text
    text = re.sub(r'[^\w\s.,?!:;\'-]', '', text)

    text = ' '.join(text.split())

    text = text.lower()

    return text.strip()
# Load the test data
data_test=pd.read_csv("/content/samsum-train.csv")

data_test.drop_duplicates(subset=['dialogue'],inplace=True)
data_test.dropna(axis=0,inplace=True)
stop_words = set(stopwords.words('english'))

cleaned_text = []
for t in data_test['dialogue']:
    cleaned_text.append(text_cleaner(t))
cleaned_summary = []
for t in data_test['summary']:
    cleaned_summary.append(text_cleaner(t))

data_test['cleaned_text']=cleaned_text
data_test['cleaned_summary']=cleaned_summary


text_word_count = []
summary_word_count = []

for i in data_test['cleaned_text']:
      text_word_count.append(len(i.split()))

for i in data_test['cleaned_summary']:
      summary_word_count.append(len(i.split()))

length_df = pd.DataFrame({'text':text_word_count, 'summary':summary_word_count})

length_df.hist(bins = 30)
plt.show()
max_text_len=400
max_summary_len=60
cleaned_text =np.array(data_test['cleaned_text'])
cleaned_summary=np.array(data_test['cleaned_summary'])

short_text=[]
short_summary=[]

for i in range(len(cleaned_text)):
    if(len(cleaned_summary[i].split())<=max_summary_len and len(cleaned_text[i].split())<=max_text_len):
        short_text.append(cleaned_text[i])
        short_summary.append(cleaned_summary[i])

df_train=pd.DataFrame({'text':short_text,'summary':short_summary})
df_train.to_csv('samsum_test_cleaned.csv', index=False)