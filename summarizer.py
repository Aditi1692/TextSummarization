from __future__ import division

from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn import metrics
from nltk.tokenize import sent_tokenize, wordpunct_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
from collections import Counter

import json
import math

OUTPUT_FILE = "summarizer.html"
EVALUATION_METRIC = "L"

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
    # Get ngrams for gold standard and predcited summary
    gold_ngram_seq = list(ngrams(gold_standard_summary, n))
    pred_ngram_seq = list(ngrams(predicted_summary, n))
    # Find common ngrams
    common_ngrams = set(gold_ngram_seq).intersection(set(pred_ngram_seq))
    # Calculate ROUGE-N score
    rouge_n_score = len(common_ngrams) / len(gold_ngram_seq)

    print "ROUGE -", n, "score for sentence", sent, "is: ", rouge_n_score


# Function to calculate and print ROUGE-L score
def calculate_rouge_l_score(sent, gold_standard_summary, predicted_summary):
    # Code to find longest common subsequence
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

    print "Recall of ROUGE-L for sentence", sent, "is", recall_lcs
    print "Precision of ROUGE-L for sentence", sent, "is", precision_lcs
    print "F1 score of ROUGE-L for sentence", sent, "is", f1_lcs


if __name__ == '__main__':
    with open('data.json') as data_file:
        data = json.load(data_file)

    # stemmer = PorterStemmer()                 # You can change the stemmer here. Though I haven't used it
    lemmatizer = WordNetLemmatizer()            # Lemmatizer which shortens the words. e.g.) causes -> cause
    vectorizer = TfidfVectorizer(ngram_range=(1, 1))              # Creates vectors of each word
    # vectorizer = CountVectorizer()

    # Used for LDA. Play around with n_topics and max_iter
    lda = LatentDirichletAllocation(n_topics=1, max_iter=10, learning_method='online', learning_offset=50.,
                                    random_state=0)

    write_output = open(OUTPUT_FILE, 'w')
    write_output.write('<html><head><title>Text Summary</title></head> <body><table border="1">')

    # Create train and test data split - 80% train and 20% test data
    train_data = data[0:int(len(data)*0.8)]
    test_data = data[int(len(data)*0.8):]

    train_data = preprocess_data(train_data)
    test_data = preprocess_data(test_data)

    for sent in train_data:
        line = train_data[sent]["words"]
        gold_standard_summary = train_data[sent]["gold_standard"]
        # Perform LDA if there are any words that are lemmatized. Skip if the content is blank
        if len(line) > 1:
            # Create a vector of each word. It will be used to perform LDA
            vectors = vectorizer.fit_transform(line)

            # Perform LDA on these word vectors
            lda.fit(vectors)

            # Get all the words in the given line
            tf_feature_names = vectorizer.get_feature_names()

            # Calculate the ROUGE-1 score
            predicted_summary = print_top_words(lda, tf_feature_names, 100, train_data[sent]["words_dict"]).\
                encode('ascii', 'ignore').split()

            if EVALUATION_METRIC == 'N':
                # Find ROUGE score with unigrams and bigrams
                print "ROUGE-N is a recall based metric"
                calculate_rouge_n_score(sent, gold_standard_summary, predicted_summary, 1)
                calculate_rouge_n_score(sent, gold_standard_summary, predicted_summary, 2)
            elif EVALUATION_METRIC == 'L':
                # Find ROUGE score with longest common subsequence
                calculate_rouge_l_score(sent, gold_standard_summary, predicted_summary)
            print

            # Print the top 10 words (if there are more than 10 words). Write output in HTML file
            write_output.write('<tr><th>' + str(sent) + '</th><td>' +
                        print_top_words(lda, tf_feature_names, 100, train_data[sent]["words_dict"]) + '</td></tr>')

    # Run LDA on test data
    for sent in test_data:
        line = test_data[sent]["words"]
        gold_standard_summary = test_data[sent]["gold_standard"]
        if len(line) > 1:
            vectors_test = vectorizer.fit_transform(line)

            lda.fit(vectors_test)

            # Get all the words in the given line
            tf_feature_names = vectorizer.get_feature_names()

            # Calculate the ROUGE-1 score
            predicted_summary = print_top_words(lda, tf_feature_names, 100, test_data[sent]["words_dict"]). \
                encode('ascii', 'ignore').split()

            if EVALUATION_METRIC == 'N':
                # Find ROUGE score with unigrams and bigrams
                print "ROUGE-N is a recall based metric"
                calculate_rouge_n_score(sent, gold_standard_summary, predicted_summary, 1)
                calculate_rouge_n_score(sent, gold_standard_summary, predicted_summary, 2)
            elif EVALUATION_METRIC == 'L':
                # Find ROUGE score with longest common subsequence
                calculate_rouge_l_score(sent, gold_standard_summary, predicted_summary)
            print

            # Print the top 10 words (if there are more than 10 words). Write output in HTML file
            write_output.write('<tr><th>' + str(sent) + '</th><td>' +
                               print_top_words(lda, tf_feature_names, 100, test_data[sent]["words_dict"]) + '</td></tr>')

    write_output.write('</table></body></html')
    write_output.close()
