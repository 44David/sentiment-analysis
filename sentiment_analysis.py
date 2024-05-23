import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import tensorflow as tf
from transformers import AutoTokenizer
from transformers import TFAutoModelForSequenceClassification
from scipy.special import softmax 

df = pd.read_csv('Reviews.csv')
df = df.head(500)
MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = TFAutoModelForSequenceClassification.from_pretrained(MODEL)

example = df['Text'][50]

def polarity_scores(example):
    encoded_text = tokenizer(example, return_tensors='tf')
    output = model(**encoded_text)
    scores = output[0][0].numpy()
    scores = softmax(scores)
    scores_hashmap = {
        'neg_sentiment_val' : scores[0],
        'neu_sentiment_val' : scores[1],
        'pos_sentiment_val' : scores[2]
    }
    return scores_hashmap

results = {}
for i, row in df.iterrows():
    try:
        text = row['Text']
        ID = row['Id']
        roberta_result = polarity_scores(text)
        unpacked_results = {**roberta_result}
        results[ID] = unpacked_results
    except tf.errors.InvalidArgumentError:
        print(f'ID error at {ID}')

results_df = pd.DataFrame(results).T
results_df = results_df.reset_index().rename(columns={"index": 'Id'})
results_df = results_df.merge(df, how='left')

print(results_df.head())



