# To generate CoNLL file, type the commands:
#      java -mx4096m -cp "../../stanford-parser-full-2017-06-09/*:" edu.stanford.nlp.parser.lexparser.LexicalizedParser -sentences "newline" -maxLength "300" -outputFormat "penn" /Users/thalamus/nltk_data/englishPCFG.ser.gz extracted_opinions.txt >testsent.tree
#      java -mx4096m -cp "../../stanford-parser-full-2017-06-09/*:" edu.stanford.nlp.trees.EnglishGrammaticalStructure -treeFile testsent.tree -conllx > opinions.conll

from nltk.parse.stanford import StanfordDependencyParser

import os
import nltk

# Change this to point to your nltk_data directory
os.environ['STANFORD_PARSER'] = '/Users/thalamus/nltk_data'
os.environ['STANFORD_MODELS'] = '/Users/thalamus/nltk_data'

INPUT_FILE_PATH = 'extracted_opinions.txt'
CONLL_FILE = 'opinions.conll'