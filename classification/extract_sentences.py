import os
import json
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, wordpunct_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import networkx as nx

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

def words_in_string(a_string):
	if set(list_of_reason).intersection(a_string.split()):
		return 'reason'
	elif set(list_of_justification).intersection(a_string.split()):
		return 'justification'
	elif set(list_of_outcome).intersection(a_string.split()):
		return 'outcome'
	elif set(list_of_goal).intersection(a_string.split()):
		return 'goal'
	else:
		return 'not defined'


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

import matplotlib.pyplot as plt

def build_frame(list_of_words):
	G = nx.Graph()
	for key, value in list_words.iteritems():
		if value == 'NOUN':
			G.add_node(key)
		elif value == 'VERB':
			G.add_edge(key)
	nx.draw(G)
	plt.show()




if __name__ == "__main__":
	# Get the file from the parent directory
	with open(os.path.join(os.pardir, INPUT_FILE_PATH)) as data_file:
		data = json.load(data_file)

		# stemmer = PorterStemmer()                 # You can change the stemmer here. Though I haven't used it
		lemmatizer = WordNetLemmatizer()  # Lemmatizer which shortens the words. e.g.) causes -> cause
		# vectorizer = TfidfVectorizer()              # Creates vectors of each word
		vectorizer = CountVectorizer()
		all_text = []
		# Used for LDA. Play around with n_topics and max_iter
		lda = LatentDirichletAllocation(n_topics=1, max_iter=10, learning_method='online', learning_offset=50.,
										random_state=0)
		# List to store the lemmatized words
		lemmatized_words = list()
		
	#write_output = open(OUTPUT_FILE_PATH, 'w')




	#for word in words_in_string(my_word_list, a_string):
	#	print(word)

	for line in data:
		write_output = open(OUTPUT_FILE_PATH, 'w')
		normal_words = line["content"].encode('ascii', 'ignore')
		category = words_in_string(normal_words)

		list_words = dict()
		word_order_dict = dict()
		count = 0
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
			vectors = vectorizer.fit_transform(lemmatized_words)

			# Perform LDA on these word vectors
			lda.fit(vectors)

			# Get the top words after performing LDA
			tf_feature_names = vectorizer.get_feature_names()

			# Print the top 10 words (if there are more than 10 words). Write output in HTML file
			write_output.write("%s \t" %
							   print_top_words(lda, tf_feature_names, 100, word_order_dict))
			write_output.write("%s\n" %category)

			write_output.close()

			os.system(
				"java -mx4096m -cp \"/home/aditi/Downloads/stanford-parser-full-2017-06-09/*:\" edu.stanford.nlp.parser.lexparser.LexicalizedParser -sentences \"newline\" -maxLength \"300\" -outputFormat \"penn\" /home/aditi/Downloads/englishPCFG.ser.gz ~/PycharmProjects/TextSummarization/classification/extracted_opinions.txt > testsent.tree")
			os.system(
				"java -mx4096m -cp \"/home/aditi/Downloads/stanford-parser-full-2017-06-09/*:\" edu.stanford.nlp.trees.EnglishGrammaticalStructure -treeFile testsent.tree -conllx > opinions.conll")

			with open(CONLL_FILE) as inputted_file:
				lines = inputted_file.read().splitlines()  # read input file

			for line in lines:
				dep_tree = line.split('\t')
				# print (dep_tree[3])

				# Check if there are more lines in the parse tree
				if len(dep_tree) != 1:
					if dep_tree[1] in list_of_connectives:
						flag = 1
					if dep_tree[3] == 'NOUN':
						list_words[dep_tree[1]] = 'NOUN'

					if dep_tree[3] == 'VERB':
						list_words[dep_tree[1]] = 'VERB'

			build_frame(list_words)

			lemmatized_words = list()

		#write_output.write("%s\n" % normal_words)
		




