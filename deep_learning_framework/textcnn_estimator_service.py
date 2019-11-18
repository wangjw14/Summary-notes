import tensorflow as tf
import pandas as pd
import os
from textcnn_estimator import params, DataProcessor

os.environ['CUDA_VISIBLE_DEVICES'] = '7'

path_vocab = './data/dbpedia_csv/vocab.txt'
path_test = './data/dbpedia_csv/test.csv'

processor = DataProcessor(path_vocab,'',params)

predictor = tf.contrib.predictor.from_saved_model('export_model/1574065922')

df = pd.read_csv(path_test,header=None)
df = df.sample(frac=1)[:10]
print(df)
df[2] = df[2].apply(lambda x:processor.process_text(x))

d = list(df[2].values)
print(d)

r = predictor({"text_ids":d})
print(r)
