# Text Summarization
- Aditi Jain
- Nipun Bayas

## Purpose

The aim of this project is to summarize the text present in an HTML page using different algorithms, like the Latent Dirichlet Allocation (LDA) algorithm.

In order to run the summarizer program, use the following command:

`python summarizer.py <metric>`
    
    where: metric can be N, L, S or A(all)

## Classification

We have used the Stanford Dependency Parser to create dependency trees for each opinion. Based on these trees, we aim to extract summaries, based on the words and their part of speech in the paragraph.

The code for classification can be found under classification/parsing_opinions.py

## References
* https://rstudio-pubs-static.s3.amazonaws.com/79360_850b2a69980c4488b1db95987a24867a.html
* https://en.wikipedia.org/wiki/Automatic_summarization#Evaluation_techniques
* http://www.aclweb.org/anthology/W/W04/W04-1013.pdf
* https://rare-technologies.com/text-summarization-in-python-extractive-vs-abstractive-techniques-revisited/
* http://glowingpython.blogspot.com/2014/09/text-summarization-with-nltk.html
* http://christop.club/2014/05/06/using-gensim-for-lda/
* https://arxiv.org/pdf/1509.00685.pdf
