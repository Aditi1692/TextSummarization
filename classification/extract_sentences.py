import os
import json
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, wordpunct_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import networkx as nx


class wordAttr(object):
	def __init__(self, word, type):
		self.word = word
		self.type = type


INPUT_FILE_PATH = "data.json"
OUTPUT_FILE_PATH = "extracted_opinions.txt"
CONLL_FILE = 'opinions.conll'

list_of_connectives = ['accordingly', 'additionally', 'after', 'afterward', 'afterwards', 'also', 'alternatively',
                           'although', 'and', 'as', 'as a result', 'as an alternative', 'as if', 'as long as',
                           'as soon as', 'as though', 'as well', 'because', 'before', 'before and after', 'besides',
                           'but', 'by comparison', 'by contrast', 'by then', 'consequently', 'conversely', 'earlier',
                           'either', 'else', 'except', 'finally', 'for', 'for example', 'for instance', 'further',
                           'furthermore', 'hence', 'however', 'if', 'if and when', 'then', 'in addition', 'in contrast',
                           'in fact', 'in other words', 'in particular', 'in short', 'in sum', 'in the end', 'in turn',
                           'indeed', 'insofar as', 'instead', 'later', 'lest', 'likewise', 'meantime', 'meanwhile',
                           'moreover', 'much as', 'neither', 'nevertheless', 'next', 'nonetheless', 'nor', 'now that',
                           'on the contrary', 'on the one hand', 'on the other hand', 'once', 'or', 'otherwise',
                           'overall', 'plus', 'previously', 'rather', 'regardless', 'separately', 'similarly',
                           'simultaneously', 'since', 'so', 'so that', 'specifically', 'still', 'then', 'thereafter',
                           'thereby', 'therefore', 'though', 'thus', 'till', 'ultimately', 'unless', 'until', 'when',
                           'when and if', 'whereas', 'while', 'yet']


list_of_reason = ['because', 'since']
list_of_justification = ['so']
list_of_goal = ['should', 'be', 'must']
list_of_reinforcement = ['need']
list_of_outcome = ['will be', 'will', 'so that', 'if', 'when', 'would']

def summarize_text(text):
	# stemmer = PorterStemmer()                 # You can change the stemmer here. Though I haven't used it
	lemmatizer = WordNetLemmatizer()  # Lemmatizer which shortens the words. e.g.) causes -> cause
	# vectorizer = TfidfVectorizer()              # Creates vectors of each word
	vectorizer = CountVectorizer(stop_words=None)
	all_text = []
	# Used for LDA. Play around with n_topics and max_iter
	lda = LatentDirichletAllocation(n_topics=1, max_iter=10, learning_method='online', learning_offset=50.,
									random_state=0)
	# List to store the lemmatized words
	lemmatized_words = list()
	list_words = dict()
	word_order_dict = dict()
	count = 0
	normal_words = ""
	summary = list()
	# Used word tokenizer to tokenize the words
	tokens = word_tokenize(normal_words)
	for token in tokens:
		token = token.lower()
		# stemmed_words.append(stemmer.stem(token))
		# Used lemmatizer on the words
		lemmatized_words.append(token)
		word_order_dict[count] = token
		count += 1

	# Removed stop words
		lemmatized_words = [word for word in lemmatized_words if word not in stopwords.words('english')]

	# print word2idx

		# Perform LDA if there are any words that are lemmatized. Skip if the content is blank
		if len(lemmatized_words) > 1:
			# Create a vector of each word. It will be used to perform LDA
			#print lemmatized_words
			vectors = vectorizer.fit_transform(lemmatized_words)

			# Perform LDA on these word vectors
			lda.fit(vectors)

			# Get the top words after performing LDA
			tf_feature_names = vectorizer.get_feature_names()
			summary.append(print_top_words(lda, tf_feature_names, 100, word_order_dict))
			lemmatized_words = list()


	return summary


# Method to print the top words from LDA
def print_top_words(model, feature_names, n_top_words, word_order_dict):
    features = list()

    # Preserve the order of words in the sentence
    for feature in feature_names:
        # Get each of the top words from LDA and convert to normal format
        feature = feature.encode('ascii', 'ignore')
        for idx, wrd in word_order_dict.iteritems():
            # compare with each word in word_order_dict dictionary. If there is a match, store the index
            if feature == wrd:
                features.append(idx)
    # Sort the indices to get the correct order of words in the sentence
    features.sort()

    return " ".join([word_order_dict[i] for i in features])

def compare_data(summary, data):
	data_list = data.split(' ')
	count = 0
	missed_word = list()
	for word in summary:
		if word != data_list[count]:
			missed_word.append(data_list[count])
		count+=1



def check_disjoint_graph(G):
	isDisjoint = False
	if isDisjoint:
		summary = summarize_text(normal_words)
		compare_data(summary, normal_words)


import matplotlib.pyplot as plt


def build_graph(graph_list):
	n1 = ""
	consecutiveNum = 0
	edge = ""
	count = 0
	frame_type = 'undefined'
	for key, value in graph_list.iteritems():
		G = nx.Graph()
		if value.type in node_list:
			if count%2 != 0 or count == 0:
				print "Adding edge: " + edge
				G.add_edge(count-1, count, label = edge, frame_type = frame_type)
				edge = ""
			if key - consecutiveNum == 1 or count == 0:
				n1+=" "+value.word
				consecutiveNum = key
			else:
				n1 = value.word
		if value.type in edge_list:
			if n1 != "":
				print "Adding node: " + n1
				G.add_node(count, label = n1)
				n1 = ""
				count += 1
			edge += value.word

		if value.word in list_of_justification:
			print "Frame Type: Justification"
			frame_type = 'justification'
			G.add_edge(1, count, label = edge, frame_type = frame_type)

		if value.word in list_of_goal:
			frame_type = 'Frame Type: goal'
			G.add_edge(1, count, label=edge, frame_type=frame_type)

		if value.word in list_of_outcome:
			frame_type = 'Frame Type: outcome'
			G.add_edge(1, count, label=edge, frame_type=frame_type)

		if value.word in list_of_reason:
			frame_type = 'Frame Type: reason'
			G.add_edge(1, count, label=edge, frame_type=frame_type)

	if n1 != "":
		print "Add node: "+ n1
		G.add_node(count, label = n1)
		count+=1

	if edge != "":
		print "Add edge"+ edge
		G.add_edge(count-1, count, label = edge, frame_type = frame_type)

	#nx.draw(G)
	#plt.draw()

	# TODO:
	#check for any disjoint graph
	check_disjoint_graph(G)


from nltk.parse.dependencygraph import DependencyGraph

if __name__ == "__main__":
	# Get the file from the parent directory
	with open(os.path.join(os.pardir, INPUT_FILE_PATH)) as data_file:
		data = json.load(data_file)

	index = 0
	for line in data:
		write_output = open(OUTPUT_FILE_PATH, 'w')
		normal_words = line["content"].encode('ascii', 'ignore')
		write_output.write(normal_words)

	write_output.close()
	#category = words_in_string(normal_words)
	#construct_graph(data)

	os.system(
		"java -mx4096m -cp \"/home/aditi/Downloads/stanford-parser-full-2017-06-09/*:\" edu.stanford.nlp.parser.lexparser.LexicalizedParser -sentences \"newline\" -maxLength \"300\" -outputFormat \"penn\" /home/aditi/Downloads/englishPCFG.ser.gz ~/PycharmProjects/TextSummarization/classification/extracted_opinions.txt > testsent.tree")
	os.system(
		"java -mx4096m -cp \"/home/aditi/Downloads/stanford-parser-full-2017-06-09/*:\" edu.stanford.nlp.trees.EnglishGrammaticalStructure -treeFile testsent.tree -conllx > opinions.conll")

	with open(CONLL_FILE) as inputted_file:
		lines = inputted_file.read().splitlines()  # read input file

	node_list = ['NN', 'NNS', 'NNP', 'NNPS', 'PRP', 'JJ']
	edge_list = ['VB', 'VBN', 'VBD', 'VBG', 'VBP', 'VBZ']

	join_node = dict()
	# store the word with its part of speech
	graph_list = dict()

	part_of_speech = dict()
	for line in lines:
		dep_tree = line.split('\t')
		#print dep_tree

		if dep_tree[0] != '':
			if len(dep_tree) > 1:
				part_of_speech[dep_tree[0]] = dep_tree[4]
				# check if it's a noun or verb or any kind of frame
				if dep_tree[4] in node_list or dep_tree[4] in edge_list \
						or dep_tree[1] in list_of_justification\
						or dep_tree[1] in list_of_reason\
						or dep_tree[1] in list_of_outcome\
						or dep_tree[1] in list_of_goal:
					attr = {'word': dep_tree[1], 'type': dep_tree[4]}
					#print dep_tree[1] + dep_tree[4]
					value = wordAttr(**attr)
					graph_list[index] = value
					index += 1
		#print dep_tree[0]
		else:
			print "New sentence"
			build_graph(graph_list)
			graph_list = dict()
			index = 0














