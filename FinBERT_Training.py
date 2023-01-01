import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification, pipeline

# based upon article: 
# https://wandb.ai/ivangoncharov/FinBERT_Sentiment_Analysis_Project/reports/Financial-Sentiment-Analysis-on-Stock-Market-Headlines-With-FinBERT-Hugging-Face--VmlldzoxMDQ4NjM0

# inspired by Huggingface user IoannisTr and his Tech_Stocks_Trading_Assistant
# https://huggingface.co/spaces/IoannisTr/Tech_Stocks_Trading_Assistant
# https://github.com/yya518/FinBERT/blob/master/FinBERT-demo.ipynb

# loading in entire headline dataset from Kaggle 
# https://www.kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests?resource=download

class FinBERT():
    def __init__(self):
        self.headlines = pd.read_csv('../Headline_Data/raw_partner_headlines.csv')
        self.np_headlines = np.array(self.headlines)
        np.random.shuffle(self.np_headlines)

        self.tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
        self.model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone', num_labels=3)

        self.data_pipeline = pipeline("sentiment-analysis", 
                                       model=self.model,
                                       tokenizer = self.tokenizer)
    def input(self, sentence):
        return self.data_pipeline(sentence)

