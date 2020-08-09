from absl import logging

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import tensorflow_hub as hub
import sentencepiece as spm
from sklearn import manifold, datasets
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
import time
import random
import string
import itertools
import json
from pathlib import Path, PureWindowsPath
base_path = Path(__file__).parent
module_path = str(PureWindowsPath(base_path / "/lite_encoder")).replace("\\","/")		
module = hub.Module("C:/Users/New/OneDrive/CONTRACTING/graphator/lite_encoder")
	
input_placeholder = tf.sparse_placeholder(tf.int64, shape=[None, None])
encodings = module(
		inputs=dict(
				values=input_placeholder.values,
				indices=input_placeholder.indices,
				dense_shape=input_placeholder.dense_shape))
				
with tf.Session() as sess:
	spm_path = sess.run(module(signature="spm_path"))

sp = spm.SentencePieceProcessor()
sp.Load(spm_path)
print("SentencePiece model loaded at {}.".format(spm_path))

def process_to_IDs_in_sparse_format(sp, sentences):
	# An utility method that processes sentences with the sentence piece processor
	# 'sp' and returns the results in tf.SparseTensor-similar format:
	# (values, indices, dense_shape)
	ids = [sp.EncodeAsIds(x) for x in sentences]
	max_len = max(len(x) for x in ids)
	dense_shape=(len(ids), max_len)
	values=[item for sublist in ids for item in sublist]
	indices=[[row,col] for row in range(len(ids)) for col in range(len(ids[row]))]
	return (values, indices, dense_shape)

#GENERATE DUMMY DATA - remove this later when an API
texts = [
	# Smartphones
	"I like my phone",
	"My phone is not good.",
	"Your cellphone looks great.",
	"Hack the Android system",
	"I am learning to code a program",
	"The hard drive and battery need a repair",

	# Weather
	"Will it snow tomorrow?",
	"Recently a lot of hurricanes have hit the US",
	"Global warming is real",
	"Many clouds form high the sky",
	"Police search for missing kayaker, Bungendore flooded, warnings for Queanbeyan and Oaks Estate as rainfall hits region"
	
	# Business
	"The recruitment team worked on your contract",
	"HR fired him at his desk",
	"The boss hates office politics",
	"He sued for compensation",
	"Our profits this quarter look great",
	"Apple stocks were sold high",
	"Japan unsuccessful in lifting auto tariffs early in UK trade deal: media"

	# Food and health
	"An apple a day, keeps the doctors away",
	"Eating strawberries is healthy",
	"Is paleo better than keto?",
	"Cheese is better for you than red meat",
	"Pandemic defies treatments",
	"I look to cook at least twice a week",

	# Asking about age
	"How old are you?",
	"what is your age?",
	"Retirement planning is crucial",
	"The elderly often have health problems",
	
	#Politics
	"He voted for the best politican",
	"The Republican Party has problems",
	"The corruption scandal rocked the government",
	"Party is looking bad in the polls",
	"Belarus holds election as street protests rattle strongman president",
	"Esper: U.S. will cut troop levels in Afghanistan to 'less than 5,000'",
	
	#Longer texts,
	"Victoria, at the centre of a second wave of infections in Australia, reported 394 cases of the novel coronavirus in the past 24 hours, compared with a daily average of 400-500 over the past week. The new deaths bring the state’s total to 210.",
	"Eastman Kodak Co’s $765 million loan agreement with the U.S. government to produce pharmaceutical ingredients has been put on hold due to “recent allegations of wrongdoing,” the U.S. International Development Finance Corp (DFC) said."
	
]
letters = string.ascii_lowercase
docs=[]
for doc in texts:
	id =''.join(random.choice(letters) for i in range(10)) 
	dooc = {'_id':id, 'text':doc}
	docs.append(dooc)

# END GENERATING DUMMY DATA	

#GET SENTENCE EMBEDDINGS	
#make arrays of the text (to turn into embedings ) and id (to map back to texts concisely)
fulltexts = [d['text'] for d in docs]
doc_ids = [d['_id'] for d in docs]	


# Reduce logging output.
logging.set_verbosity(logging.ERROR)
session = tf.Session()
session.run([tf.global_variables_initializer(), tf.tables_initializer()])
values, indices, dense_shape = process_to_IDs_in_sparse_format(sp,texts)

embeddings_vectors = session.run(
	encodings,
	feed_dict={input_placeholder.values: values, input_placeholder.indices: indices,input_placeholder.dense_shape: dense_shape}
	)

#REDUCE DIMENSIONALITY OF EMBEDDINGS TO 2-DIMENSIONS
#number of target dimensions
n_samples = len(doc_ids)
n_components = 2
#how focused we are on the local vs. global - needs to increase with no. samples
perplexity_alpha = 2+round((n_samples*0.1),0)
perplexities = [3,perplexity_alpha]
#learning rate - 10-1000, too low and data stays dense, too high and circle of equidistant points
learning_rate_alpha = max(15,n_samples/12)
learning_rate = 30

# get 2D coordinates for each embedding
tsne = manifold.TSNE(n_components=n_components, learning_rate=learning_rate_alpha, init='random',
					 random_state=0, perplexity=perplexity_alpha, n_iter=5000, n_iter_without_progress=2000)
scatter_plot_points = tsne.fit_transform(embeddings_vectors)

listpoints = scatter_plot_points.tolist()

outputjson=[]
for points, label,text in zip(listpoints, doc_ids,fulltexts):
	jsondoc = {"coords":points,"doc_id":label,"doc_text":text}
	outputjson.append(jsondoc)

with open('exampledata.json','w',encoding='utf-8') as f:
	json.dump(outputjson, f, ensure_ascii=False, indent=4)
	
#PLOT THE GRAPH
(fig, subplots) = plt.subplots(1, 2, figsize=(30, 30))

hitme = 0
for perplexity in perplexities:
	ax = subplots[hitme]
	t0 = time.time()
	tsne = manifold.TSNE(n_components=n_components, learning_rate=learning_rate, init='random',
						 random_state=0, perplexity=perplexity, n_iter=5000, n_iter_without_progress=2000)
	scatter_plot_points = tsne.fit_transform(embeddings_vectors)
	t1 = time.time()
	print("Perplexity %s: %.2g sec" % (perplexity, t1 - t0))
	x_axis = [o[0] for o in scatter_plot_points]
	y_axis = [o[1] for o in scatter_plot_points]
	ax.scatter(x_axis, y_axis)
	ax.set_title("Perplexity=%d" % perplexity)

	for i, txt in enumerate(texts):
		ax.annotate(txt, (x_axis[i], y_axis[i]))
	hitme+=int(1)
plt.show()