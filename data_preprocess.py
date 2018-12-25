import os
import imutils 
from imutils import paths
import numpy as np
import re
import json
import io

intents_list = ['AddToPlaylist', 'BookRestaurant', 'GetWeather', 'SearchCreativeWork', 'SearchScreeningEvent' ]

BASE_DIR = ''
DATA_DIR = BASE_DIR + 'data'
EMBEDDING_DIR = BASE_DIR + '/embedding'

# Function for text preprocessing
def clean_str(string): 
	"""
	Tokenization/string cleaning for all datasets except for SST.
	Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
	""" 
	string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
	string = re.sub(r"\'s", " \'s", string)
	string = re.sub(r"\'ve", " \'ve", string)
	string = re.sub(r"n\'t", " n\'t", string)
	string = re.sub(r"\'re", " \'re", string)
	string = re.sub(r"\'d", " \'d", string)
	string = re.sub(r"\'ll", " \'ll", string)
	string = re.sub(r",", " , ", string)
	string = re.sub(r"!", " ! ", string)
	string = re.sub(r"\(", " \( ", string)
	string = re.sub(r"\)", " \) ", string)
	string = re.sub(r"\?", " \? ", string)
	string = re.sub(r"\s{2,}", " ", string)
	return string.strip().lower()

def process_training_data():

	train_req_list = []
	train_label_list = []
	
	for intent in intents_list:
		
		with io.open(os.path.join(DATA_DIR, "/train_" + intent + "_full.json.txt", encoding='utf-8')) as f:
			data = json.load(f)
		
			req_list = []
			label_list = []

			for request in data[intent]:
				txt = ''

				for text in request['data']:
					txt += text['text']
				
				txt = clean_str(txt)
				
				req_list.append(txt)
				label_list.append(intent)
		
		if not os.path.exists(DATA_DIR + "/processed_data"):
			os.mkdir(DATA_DIR + "/processed_data")

		with io.open(DATA_DIR + "/processed_data/train_" + intent + ".txt", mode='w', encoding='utf-8') as outfile:
			for req in req_list:
				outfile.write("%s\n" % req)
			
		train_req_list.extend(req_list)
		train_label_list.extend(label_list)

		test_req_list = []
		test_label_list = []

		with io.open(DATA_DIR + "/validate_" + intent + ".json.txt", encoding='utf-8') as f:
			data = json.load(f)
		
			req_list = []
			label_list = []

			for request in data[intent]:
				txt = ''

				for text in request['data']:
					txt += text['text']
				
				txt = clean_str(txt)
				
				req_list.append(txt)
				label_list.append(intent)

		if not os.path.exists(DATA_DIR + "/processed_data"):
			os.mkdir(DATA_DIR + "/processed_data")
			
		with io.open(DATA_DIR + "/processed_data/validate_" + intent + ".txt", mode='w', encoding='utf-8') as outfile:
			for req in req_list:
				outfile.write("%s\n" % req)
	
		test_req_list.extend(req_list)
		test_label_list.extend(label_list)

		np.array(train_req_list).dump(open('data/train_text.npy', 'wb'))
		np.array(train_label_list).dump(open('data/train_label.npy', 'wb'))
		np.array(test_req_list).dump(open('data/test_text.npy', 'wb'))
		np.array(test_label_list).dump(open('data/test_label.npy', 'wb'))

process_training_data()
			