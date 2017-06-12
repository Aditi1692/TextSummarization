# The results can be found at file:///C:/Users/Nipun/Desktop/MS/Summer%202017/TextSummarization/metrics.html

from __future__ import division

from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn import metrics
from nltk.tokenize import sent_tokenize, wordpunct_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams, skipgrams

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.nlp.stemmers import Stemmer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.utils import get_stop_words

import operator as op
import json
import sys

OUTPUT_FILE = "summarizer.html"
LANGUAGE = "english"
SENTENCES_COUNT = 100
METRICS_SCORE_FILE = "metrics.html"
output_scores = dict()


def preprocess_data(data):
    # Dict to store preprocessed data
    preprocessed_data = dict()

    # List to store the lemmatized words
    lemmatized_words = list()

    # Dictionary to store the order of words
    word_order_dict = dict()
    count = 0

    for line in data:
        # Changed from unicode to string
        normal_words = line["content"].encode('ascii', 'ignore')
        gold_standard = line["contentSimp"].encode('ascii', 'ignore').split()

        # Used word tokenizer to tokenize the words
        tokens = word_tokenize(normal_words)
        for token in tokens:
            token = token.lower()
            # stemmed_words.append(stemmer.stem(token))
            # Used lemmatizer on the words
            lemmatized_words.append(lemmatizer.lemmatize(token))

        # Removed stop words
        lemmatized_words = [word for word in lemmatized_words if word not in stopwords.words('english')]
        for word in lemmatized_words:
            word_order_dict[count] = lemmatizer.lemmatize(word)
            count += 1

        preprocessed_data[line["index"]] = {"words": lemmatized_words, "words_dict": word_order_dict,
                                            "gold_standard": gold_standard}

        # Reset the values for next line
        lemmatized_words = list()
        word_order_dict = dict()
        count = 0

    return preprocessed_data


def perform_sumy_summarization(data):
    stemmer = Stemmer(LANGUAGE)

    summarizers = [LsaSummarizer(stemmer), LexRankSummarizer(stemmer), TextRankSummarizer(stemmer)]

    # print "SUMY Scores: "
    # Read each sentence from 'data' and create a summary on it
    for line in data:
        # Only consider the content part of the text. Changed it from unicode to normal string
        # summarized_text = line["content"].encode('ascii', 'ignore')
        summarized_text = line["content"]
        gold_standard = line["contentSimp"]

        # Read line by line instead of reading the entire file
        parser = PlaintextParser.from_string(summarized_text, Tokenizer(LANGUAGE))

        for summarizer in summarizers:
            # Store the scores in a dictionary
            output_scores[line["index"]] = []
            summarizer.stop_words = get_stop_words(LANGUAGE)
            # print "SUMY with", summarizer
            for sentence in summarizer(parser.document, SENTENCES_COUNT):
                if int(line["index"]) in output_scores:
                    output_scores[line["index"]] = []
                # Store output in a dictionary in the form of a key-value pair
                # Example -->  1: 'with the exception of the elderly and the youth'
                output_scores[int(line["index"])].append({"sumy_rouge_unigrams":
                                        calculate_rouge_n_score(line["index"], gold_standard, str(sentence), 1)})
                output_scores[int(line["index"])].append({"sumy_rouge_bigrams":
                                        calculate_rouge_n_score(line["index"], gold_standard, str(sentence), 2)})
                output_scores[int(line["index"])].append({"sumy_rouge_l":
                                        calculate_rouge_l_score(line["index"], gold_standard, str(sentence))})
                output_scores[int(line["index"])].append({"sumy_rouge_s":
                                        calculate_rouge_s_score(line["index"], gold_standard,str(sentence), 2)})
                # print '*' * 70


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

    # for topic_idx, topic in enumerate(model.components_):
        # print("Topic #%d:" % topic_idx)
        # for i in topic.argsort()[:-n_top_words - 1:-1]:
        #     features.append(i)
        # return (" ".join([feature_names[i]
        #                 for i in features]))


# Function to calculate and print ROUGE-N score
def calculate_rouge_n_score(sent, gold_standard_summary, predicted_summary, n):
    # Get ngrams for gold standard and predicted summary
    gold_ngram_seq = list(ngrams(gold_standard_summary, n))
    pred_ngram_seq = list(ngrams(predicted_summary, n))
    # Find common ngrams
    common_ngrams = set(gold_ngram_seq).intersection(set(pred_ngram_seq))
    # Calculate ROUGE-N score
    rouge_n_score = len(common_ngrams) / len(gold_ngram_seq)

    # print "ROUGE -", n, "score for sentence", sent, "is: ", rouge_n_score
    return ['NA', round(rouge_n_score, 4), 'NA']


# Function to calculate and print ROUGE-L score
def calculate_rouge_l_score(sent, gold_standard_summary, predicted_summary):
    # Code to find longest common sub-sequence
    table = [[0] * (len(predicted_summary) + 1) for _ in xrange(len(gold_standard_summary) + 1)]
    for i, ca in enumerate(gold_standard_summary, 1):
        for j, cb in enumerate(predicted_summary, 1):
            table[i][j] = (
                table[i - 1][j - 1] + 1 if ca == cb else
                max(table[i][j - 1], table[i - 1][j]))
    lcs_length = table[-1][-1]
    recall_lcs = lcs_length/len(gold_standard_summary)
    precision_lcs = lcs_length / len(predicted_summary)

    beta_value = precision_lcs/ recall_lcs
    f1_lcs = ((1 + (beta_value ** 2)) * recall_lcs * precision_lcs) / (recall_lcs + ((beta_value ** 2) * precision_lcs))

    # print "Recall of ROUGE-L for sentence", sent, "is", recall_lcs
    # print "Precision of ROUGE-L for sentence", sent, "is", precision_lcs
    # print "F1 score of ROUGE-L for sentence", sent, "is", f1_lcs
    return [round(precision_lcs, 4), round(recall_lcs, 4), round(f1_lcs, 4)]


# Function to calculate and print ROUGE-S score
def calculate_rouge_s_score(sent, gold_standard_summary, predicted_summary, n):
    gold_skipgrams = list(skipgrams(gold_standard_summary, n, n))
    pred_skipgrams = list(skipgrams(predicted_summary, n, n))

    # Find common skipgrams
    common_skipgrams = set(gold_skipgrams).intersection(set(pred_skipgrams))

    # Find mC2 for calculating the precision, recall and F1 score
    r = min(2, len(gold_standard_summary) - 2)
    if r == 0:
        return 1
    numerator = reduce(op.mul, xrange(len(gold_standard_summary), len(gold_standard_summary) - r, -1))
    denominator = reduce(op.mul, xrange(1, r + 1))
    gold_skipgram_combinations = numerator // denominator

    # Find nC2 for calculating the precision, recall and F1 score
    r = min(2, len(predicted_summary) - 2)
    if r == 0:
        return [1, 1, 1]
    numerator = reduce(op.mul, xrange(len(predicted_summary), len(predicted_summary) - r, -1))
    denominator = reduce(op.mul, xrange(1, r + 1))
    pred_skipgram_combinations = numerator // denominator

    recall_skipgram = len(common_skipgrams) / gold_skipgram_combinations
    precision_skipgram = len(common_skipgrams) / pred_skipgram_combinations

    beta_value = precision_skipgram / recall_skipgram       # Or beta can be hardcoded as 1
    f1_skipgram = ((1 + (beta_value ** 2)) * recall_skipgram * precision_skipgram) / (recall_skipgram + ((beta_value ** 2) *
                                                                                                    precision_skipgram))
    # print "Recall of ROUGE-S for sentence", sent, "is", recall_skipgram
    # print "Precision of ROUGE-S for sentence", sent, "is", precision_skipgram
    # print "F1 score of ROUGE-S for sentence", sent, "is", f1_skipgram
    return [round(precision_skipgram, 4), round(recall_skipgram, 4), round(f1_skipgram, 4)]


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("ERROR: Invalid number of arguments. \nUSAGE: summarizer.py < Evaluation_Metric ['N', 'L', 'S', 'A'] >")
        sys.exit()

    if sys.argv[1] not in ['N', 'L', 'S', 'A']:
        print("ERROR: Incorrect Metric specified. Use either 'N', 'L', 'S' or 'A")
        sys.exit()

    EVALUATION_METRIC = sys.argv[1]

    with open('data.json') as data_file:
        data = json.load(data_file)

    # stemmer = PorterStemmer()                 # You can change the stemmer here. Though I haven't used it
    lemmatizer = WordNetLemmatizer()            # Lemmatizer which shortens the words. e.g.) causes -> cause
    vectorizer = TfidfVectorizer(ngram_range=(1, 1))              # Creates vectors of each word
    # vectorizer = CountVectorizer()

    # Run SUMY and note the output
    perform_sumy_summarization(data)

    # Used for LDA. Play around with n_topics and max_iter
    lda = LatentDirichletAllocation(n_topics=1, max_iter=10, learning_method='online', learning_offset=50.,
                                    random_state=0)

    write_output = open(OUTPUT_FILE, 'w')
    write_output.write('<html><head><title>Text Summary</title></head> <body><table border="1">')

    write_metrics_output = open(METRICS_SCORE_FILE, 'w')
    write_metrics_output.write('<html><head><title>Metrics Output</title><style>td, th { padding:10px 5px; }</style></head> <body>')
    write_metrics_output.write('<table align="center" border="1" style="font-family:Arial, sans-serif;">')
    write_metrics_output.write('<tr style="padding:10px 5px;"><th rowspan="2">Sentence Index<br></th><th rowspan="2">Metric<br></th>')
    write_metrics_output.write('<th colspan="3">SUMY</th>')
    write_metrics_output.write('<th colspan="3">LDA</th></tr>')

    write_metrics_output.write('<tr align="center"><td><b>Precision</b></td><td><b>Recall</b></td><td><b>F1 Score</b></td>')
    write_metrics_output.write('<td><b>Precision</b></td><td><b>Recall</b></td><td><b>F1 Score</b></td></tr>')

    # Create train and test data split - 80% train and 20% test data
    train_data = data[0:int(len(data)*0.8)]
    test_data = data[int(len(data)*0.8):]

    train_data = preprocess_data(train_data)
    test_data = preprocess_data(test_data)
    processed_data = preprocess_data(data)

    # print "LDA Values:"
    for sent in processed_data:
        line = processed_data[sent]["words"]
        gold_standard_summary = processed_data[sent]["gold_standard"]
        # Perform LDA if there are any words that are lemmatized. Skip if the content is blank
        if len(line) > 1:
            # Create a vector of each word. It will be used to perform LDA
            vectors = vectorizer.fit_transform(line)

            # Perform LDA on these word vectors
            lda.fit(vectors)

            # Get all the words in the given line
            tf_feature_names = vectorizer.get_feature_names()

            predicted_summary = print_top_words(lda, tf_feature_names, 100, processed_data[sent]["words_dict"]).\
                encode('ascii', 'ignore').split()

            if EVALUATION_METRIC == 'N':
                # Find ROUGE-N score with unigrams and bigrams
                # print "ROUGE-N is a recall based metric"
                output_scores[int(sent)].append({"lda_rouge_unigrams":
                                        calculate_rouge_n_score(sent, gold_standard_summary, predicted_summary, 1)})
                output_scores[int(sent)].append({"lda_rouge_bigrams":
                                        calculate_rouge_n_score(sent, gold_standard_summary, predicted_summary, 2)})
            elif EVALUATION_METRIC == 'L':
                # Find ROUGE-L score with longest common subsequence
                output_scores[int(sent)].append({"lda_rouge_l":
                                        calculate_rouge_l_score(sent, gold_standard_summary, predicted_summary)})
            elif EVALUATION_METRIC == 'S':
                # Find ROUGE-S score with skip bigram co-occurrence statistics
                output_scores[int(sent)].append({"lda_rouge_s":
                                        calculate_rouge_s_score(sent, gold_standard_summary, predicted_summary, 2)})
            elif EVALUATION_METRIC == 'A':
                # Find all ROUGE scores for the given sentence
                output_scores[int(sent)].append({"lda_rouge_unigrams":
                                        calculate_rouge_n_score(sent, gold_standard_summary, predicted_summary, 1)})
                output_scores[int(sent)].append({"lda_rouge_bigrams":
                                        calculate_rouge_n_score(sent, gold_standard_summary, predicted_summary, 2)})
                # print '-' * 60
                output_scores[int(sent)].append({"lda_rouge_l":
                                        calculate_rouge_l_score(sent, gold_standard_summary, predicted_summary)})
                # print '-' * 60
                output_scores[int(sent)].append({"lda_rouge_s":
                                        calculate_rouge_s_score(sent, gold_standard_summary, predicted_summary, 2)})
            # print '*' * 70
            # print

            # Print the top 10 words (if there are more than 10 words). Write output in HTML file
            write_output.write('<tr><th>' + str(sent) + '</th><td>' +
                        print_top_words(lda, tf_feature_names, 100, processed_data[sent]["words_dict"]) + '</td></tr>')

    # # Run LDA on test data
    # for sent in test_data:
    #     line = test_data[sent]["words"]
    #     gold_standard_summary = test_data[sent]["gold_standard"]
    #     if len(line) > 1:
    #         vectors_test = vectorizer.fit_transform(line)
    #
    #         lda.fit(vectors_test)
    #
    #         # Get all the words in the given line
    #         tf_feature_names = vectorizer.get_feature_names()
    #
    #         # Calculate the ROUGE-1 score
    #         predicted_summary = print_top_words(lda, tf_feature_names, 100, test_data[sent]["words_dict"]). \
    #             encode('ascii', 'ignore').split()
    #
    #         if EVALUATION_METRIC == 'N':
    #             # Find ROUGE-N score with unigrams and bigrams
    #             # print "ROUGE-N is a recall based metric"
    #             calculate_rouge_n_score(sent, gold_standard_summary, predicted_summary, 1)
    #             calculate_rouge_n_score(sent, gold_standard_summary, predicted_summary, 2)
    #         elif EVALUATION_METRIC == 'L':
    #             # Find ROUGE-L score with longest common subsequence
    #             calculate_rouge_l_score(sent, gold_standard_summary, predicted_summary)
    #         elif EVALUATION_METRIC == 'S':
    #             # Find ROUGE-S score with skip bigram co-occurrence statistics
    #             calculate_rouge_s_score(sent, gold_standard_summary, predicted_summary, 2)
    #         elif EVALUATION_METRIC == 'A':
    #             # Find all ROUGE scores for the given sentence
    #             # print "SENTENCE", sent
    #             calculate_rouge_n_score(sent, gold_standard_summary, predicted_summary, 1)
    #             calculate_rouge_n_score(sent, gold_standard_summary, predicted_summary, 2)
    #             # print '-' * 60
    #             calculate_rouge_l_score(sent, gold_standard_summary, predicted_summary)
    #             # print '-' * 60
    #             calculate_rouge_s_score(sent, gold_standard_summary, predicted_summary, 2)
    #         # print '*' * 70
    #         # print
    #
    #         # Print the top 10 words (if there are more than 10 words). Write output in HTML file
    #         write_output.write('<tr><th>' + str(sent) + '</th><td>' +
    #                        print_top_words(lda, tf_feature_names, 100, test_data[sent]["words_dict"]) + '</td></tr>')

    for score in output_scores:
        if not output_scores[score]:
            continue

        write_metrics_output.write('<tr align="center"> <td rowspan="4">' + str(score) + '</td><td>Rouge Unigrams</td><td>' +
                                   str(output_scores[score][0]["sumy_rouge_unigrams"][0]) +
                                   '</td><td>' + str(output_scores[score][0]["sumy_rouge_unigrams"][1]) +
                                   '</td><td>' + str(output_scores[score][0]["sumy_rouge_unigrams"][2]) +
                                   '</td><td>' + str(output_scores[score][4]['lda_rouge_unigrams'][0]) +
                                   '</td><td>' + str(output_scores[score][4]['lda_rouge_unigrams'][1]) +
                                   '</td><td>' + str(output_scores[score][4]['lda_rouge_unigrams'][2]) + '</td></tr>')
        write_metrics_output.write('<tr align="center"><td>Rouge Bigrams</td><td>' +
                                   str(output_scores[score][1]["sumy_rouge_bigrams"][0]) +
                                   '</td><td>' + str(output_scores[score][1]["sumy_rouge_bigrams"][1]) +
                                   '</td><td>' + str(output_scores[score][1]["sumy_rouge_bigrams"][2]) +
                                   '</td><td>' + str(output_scores[score][5]['lda_rouge_bigrams'][0]) +
                                   '</td><td>' + str(output_scores[score][5]['lda_rouge_bigrams'][1]) +
                                   '</td><td>' + str(output_scores[score][5]['lda_rouge_bigrams'][2]) + '</td></tr>')
        write_metrics_output.write('<tr align="center"><td>Rouge L</td><td>' +
                                   str(output_scores[score][2]["sumy_rouge_l"][0]) +
                                   '</td><td>' + str(output_scores[score][2]["sumy_rouge_l"][1]) +
                                   '</td><td>' + str(output_scores[score][2]["sumy_rouge_l"][2]) +
                                   '</td><td>' + str(output_scores[score][6]['lda_rouge_l'][0]) +
                                   '</td><td>' + str(output_scores[score][6]['lda_rouge_l'][1]) +
                                   '</td><td>' + str(output_scores[score][6]['lda_rouge_l'][2]) + '</td></tr>')
        write_metrics_output.write('<tr align="center"><td>Rouge S</td><td>' +
                                   str(output_scores[score][3]["sumy_rouge_s"][0]) +
                                   '</td><td>' + str(output_scores[score][3]["sumy_rouge_s"][1]) +
                                   '</td><td>' + str(output_scores[score][3]["sumy_rouge_s"][2]) +
                                   '</td><td>' + str(output_scores[score][7]['lda_rouge_s'][0]) +
                                   '</td><td>' + str(output_scores[score][7]['lda_rouge_s'][1]) +
                                   '</td><td>' + str(output_scores[score][7]['lda_rouge_s'][2]) + '</td></tr>')

    write_output.write('</table></body></html')
    write_output.close()
    write_metrics_output.write('</table></body></html>')
    write_metrics_output.close()
