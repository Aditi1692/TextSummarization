# To generate CoNLL file, type the commands:
#      java -mx4096m -cp "../../stanford-parser-full-2017-06-09/*:" edu.stanford.nlp.parser.lexparser.LexicalizedParser -sentences "newline" -maxLength "300" -outputFormat "penn" /Users/thalamus/nltk_data/englishPCFG.ser.gz extracted_opinions.txt >testsent.tree
#      java -mx4096m -cp "../../stanford-parser-full-2017-06-09/*:" edu.stanford.nlp.trees.EnglishGrammaticalStructure -treeFile testsent.tree -conllx > opinions.conll

from nltk.parse.stanford import StanfordDependencyParser

import os
import nltk
import sys

# Change this to point to your nltk_data directory
os.environ['STANFORD_PARSER'] = '/Users/thalamus/nltk_data'
os.environ['STANFORD_MODELS'] = '/Users/thalamus/nltk_data'

INPUT_FILE = 'extracted_opinions.txt'
CONLL_FILE = 'opinions.conll'

if __name__ == '__main__':
	list_of_connectives = ['accordingly','additionally','after','afterward', 'afterwards','also','alternatively','although','and','as','as a result','as an alternative','as if','as long as','as soon as','as though','as well','because','before','before and after','besides','but','by comparison','by contrast','by then','consequently','conversely','earlier','either','else','except','finally','for','for example','for instance','further','furthermore','hence','however','if','if and when','then','in addition','in contrast','in fact','in other words','in particular','in short','in sum','in the end','in turn','indeed','insofar as','instead','later','lest','likewise','meantime','meanwhile','moreover','much as','neither','nevertheless','next','nonetheless','nor','now that','on the contrary','on the one hand','on the other hand','once','or','otherwise','overall','plus','previously','rather','regardless','separately','similarly','simultaneously','since','so','so that','specifically','still','then','thereafter','thereby','therefore','though','thus','till','ultimately','unless','until','when','when and if','whereas','while','yet']
	sentence_index = 0

	known_connectives = list()
	unknown_connectives = list()
	flag = 0
	
	with open(CONLL_FILE) as inputted_file:
		lines = inputted_file.read().splitlines() # read input file

	for line in lines:
		dep_tree = line.split('\t')

		# Check if there are more lines in the parse tree
		if len(dep_tree) != 1:
			if dep_tree[1] in list_of_connectives:
				flag = 1
		# The parse tree has ended
		else:
			if flag == 1:
				known_connectives.append(sentence_index)
			else:
				unknown_connectives.append(sentence_index)
			sentence_index += 1
			flag = 0

			# Skip the empty sentence. This should be improved
			# if sentence_index == 51:
			# 	sentence_index += 1

	print "The opinions with a connective are: "
	for opinion_id in known_connectives:
		print opinion_id

	print
	print "The opinions without a connective are: "
	for opinion_id in unknown_connectives:
		print opinion_id
