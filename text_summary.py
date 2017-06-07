from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.nlp.stemmers import Stemmer
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.utils import get_stop_words
from json2html import *

import json

LANGUAGE = "english"
SENTENCES_COUNT = 100
OUTPUT_FILE = "summary.html"

if __name__ == '__main__':
    # Dictionary to store json of summarized text
    input_json = dict()

    # Read json file and store it in data which is a list of dictionaries. Changes data to unicode format
    with open('data.json') as data_file:
        data = json.load(data_file)

    stemmer = Stemmer(LANGUAGE)

    summarizer = Summarizer(stemmer)
    summarizer.stop_words = get_stop_words(LANGUAGE)

    # Read each sentence from 'data' and create a summary on it
    for line in data:
        # Only consider the content part of the text. Changed it from unicode to normal string
        # summarized_text = line["content"].encode('ascii', 'ignore')
        summarized_text = line["content"]

        # Read line by line instead of reading the entire file
        parser = PlaintextParser.from_string(summarized_text, Tokenizer(LANGUAGE))

        for sentence in summarizer(parser.document, SENTENCES_COUNT):
            # Store output in a dictionary in the form of a key-value pair
            # Example -->  1: 'with the exception of the elderly and the youth'
            input_json[line["index"]] = str(sentence)

    # Creates a HTML table with the required summarized text
    output_json = json2html.convert(json=input_json)

    # Write the output in an HTML file 'summary.html'. Open this in the browser to see the output
    write_output = open(OUTPUT_FILE, 'w')
    write_output.write('<html><head><title>Text Summary</title></head> <body>')
    write_output.write(output_json)
    write_output.write('</body></html')
    write_output.close()
