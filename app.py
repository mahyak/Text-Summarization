# --------------- Libraries --------------- #

import math

from flask import Flask, render_template, request, redirect

import nltk
nltk.download('stopwords')
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize

import bs4 as bs
import urllib.request
import re

from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import numpy as np
import pandas as pd

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer 
from sumy.summarizers.lex_rank import LexRankSummarizer 



# --------------- Python Codes --------------- #

class general_process:
    def convert_link_text(self, link):
        data = urllib.request.urlopen(link)
        article = data.read()
        parsed_article = bs.BeautifulSoup(article, 'lxml')
        paragraghes = parsed_article.find_all('p')

        article_text = ''
        for p in paragraghes:
            article_text += p.text

        article_text = re.sub(r'\[[0-9]*\]', ' ', article_text)
        article_text = re.sub(r'\s+', ' ', article_text)
        #formatted_article_text = re.sub('[^a-zA-Z]', ' ', article_text )
        #formatted_article_text = re.sub(r'\s+', ' ', formatted_article_text)

        return article_text

class word_frequency:

    def createFrequencyTable(self, text_string)->dict:
        stopWords = set(stopwords.words('english'))
        words = word_tokenize(text_string)
        ps = PorterStemmer()
        freq_table = dict()
        
        for word in words:
            word = ps.stem(word)
            
            if word in stopWords:
                continue
            if word in freq_table:
                freq_table[word] += 1
            else:
                freq_table[word] = 1

        return freq_table
        
    def scoreSentences(self, sentences, freqTable) -> dict:
        sentenceValue = dict()

        for sentence in sentences:
            word_count_in_sentence_except_stop_words = 0
            
            for wordValue in freqTable:
                if wordValue in sentence.lower():
                    word_count_in_sentence_except_stop_words += 1

                    if sentence[:10] in sentenceValue:
                        sentenceValue[sentence[:10]] += freqTable[wordValue]
                    else:
                        sentenceValue[sentence[:10]] = freqTable[wordValue]

            if sentence[:10] in sentenceValue:
                sentenceValue[sentence[:10]] = sentenceValue[sentence[:10]] / word_count_in_sentence_except_stop_words

        return sentenceValue
        
    def findThreshold(self, sentence_value)->int:
        sum_values = 0
        
        for entry in sentence_value:
            sum_values += sentence_value[entry]
            
        average = (sum_values/len(sentence_value))
        
        return average

    def generateSummary(self, sentences, sentence_value, threshold):
        sentence_count = 0
        summary = ''
        
        for sentence in sentences:
            if sentence[:10] in sentence_value and sentence_value[sentence[:10]] >= threshold:
                summary += " "+sentence
                sentence_count += 1

        return summary

    def run_summarization(self, text):
        frequnet_table = self.createFrequencyTable(text)
        sentences = sent_tokenize(text)
        sentence_scores = self.scoreSentences(sentences, frequnet_table)
        threshold = self.findThreshold(sentence_scores)
        summary = self.generateSummary(sentences, sentence_scores, 1.3 * threshold)
        print(summary)
        return summary

class TF_IDF:
    def create_ferquency_matrix(self, sentences):
        freq_matrix = {}
        stopWords = set(stopwords.words("english"))
        ps = PorterStemmer()

        for sentence in sentences:
            freq_table = {}
            words = word_tokenize(sentence)
            
            for word in words:
                word = word.lower()
                word = ps.stem(word)
                
                if word in stopWords:
                    continue
                
                if word in freq_table:
                    freq_table[word] += 1
                else:
                    freq_table[word] = 1

            freq_matrix[sentence[:15]] = freq_table
        
        return freq_matrix

    def create_tf_matrix(self, freq_matrix):
        tf_matrix = {}
        
        for sentence, f_table in freq_matrix.items():
            tf_table = {}
            count_words_in_sentence = len(f_table)
            
            for word, count in f_table.items():
                tf_table[word] = count/count_words_in_sentence
            
            tf_matrix[sentence] = tf_table
        
        return tf_matrix

    def create_document_per_words(self, freq_matrix):
        word_per_doc_table = {}

        for f_table in freq_matrix.values():
            for word in f_table.keys():
                if word in word_per_doc_table:
                    word_per_doc_table[word] += 1
                else:
                    word_per_doc_table[word] = 1

        return word_per_doc_table

    def Create_idf_matrix(self, freq_matrix, count_doc_per_words, total_documents):
        idf_matrix = {}

        for sentence, f_table in freq_matrix.items():
            idf_table = {}

            for word in f_table.keys():
                idf_table[word] = math.log10(total_documents / float(count_doc_per_words[word]))

            idf_matrix[sentence] = idf_table
  
        return idf_matrix

    def create_tf_idf_matrix(self, tf_matrix, idf_matrix):
        tf_idf_matrix = {}

        for (sent1, f_table1), (f_table2) in zip(tf_matrix.items(), idf_matrix.values()):
            tf_idf_table = {}
    
            for (word1, value1), (value2) in zip(f_table1.items(), f_table2.values()):
                tf_idf_table[word1] = float(value1 * value2)
      
            tf_idf_matrix[sent1] = tf_idf_table

        return tf_idf_matrix

    def score_sentences(self, tf_idf_matrix)->dict:
        sentenceValue = {}
        
        for sent, f_table in tf_idf_matrix.items():
            total_score_per_sentence = 0
            count_words_in_sentence = len(f_table)
            
            for score in f_table.values():
                total_score_per_sentence += score
            
            sentenceValue[sent] = total_score_per_sentence / count_words_in_sentence
    
        return sentenceValue

    def find_threshold(self, sentence_value)->int:
        sum_values = 0
        
        for entry in sentence_value:
            sum_values += sentence_value[entry]
            
        threshold = (sum_values/len(sentence_value))

        return threshold

    def generate_summary(self, sentences, sentenceValue, threshold):
        sentence_count = 0
        summary = ''
        
        for sentence in sentences:
            if sentence[:15] in sentenceValue and sentenceValue[sentence[:15]] >= (threshold):
                summary += " " + sentence
                sentence_count += 1
        
        return summary

    def run_summarization(self, text):
        sentences = sent_tokenize(text)
        total_documents = len(sentences)
        
        freq_matrix = self.create_ferquency_matrix(sentences)
        tf_matrix = self.create_tf_matrix(freq_matrix)
        count_doc_per_words = self.create_document_per_words(freq_matrix)
        idf_matrix = self.Create_idf_matrix(freq_matrix, count_doc_per_words, total_documents)
        tf_idf_matrix = self.create_tf_idf_matrix(tf_matrix, idf_matrix)
        sentence_scores = self.score_sentences(tf_idf_matrix)
        threshold = self.find_threshold(sentence_scores)
        summary = self.generate_summary(sentences, sentence_scores, 1.3 * threshold)
        
        return summary

class text_rank:
    def remove_stopwords(self, sen):
        stop_words = stopwords.words('english')
        sentence_new = " ".join([i for i in sen if i not in stop_words])
        
        return sentence_new

    def generate_summary_rank(self, text):
        sentences = sent_tokenize(text)
        clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")
        clean_sentences = [s.lower() for s in clean_sentences]
        clean_sentences = [self.remove_stopwords(r.split()) for r in clean_sentences]

        word_embeddings = {}
        f = open('glove/glove.6B.100d.txt', encoding='utf-8')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            word_embeddings[word] = coefs
        f.close()

        sentences_vectors = []

        for sentence in clean_sentences:
            if len(sentence) != 0:
                vector = sum([word_embeddings.get(word, np.zeros((100,))) for word in sentence.split()])/(len(sentence.split())+0.001)
            else:
                vector = np.zeros((100,))
                
            sentences_vectors.append(vector)

        similarity_matrix = np.zeros([len(sentences), len(sentences)])
        
        for i in range(len(sentences)):
            for j in range(len(sentences)):
                if i != j:
                    similarity_matrix[i][j] = cosine_similarity(sentences_vectors[i].reshape(1,100), sentences_vectors[j].reshape(1,100))[0,0]

        nx_graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(nx_graph)

        ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
        summary = ''
        
        for i in range(10):
            summary += " "+ranked_sentences[i][1]

        return summary


class lex_rank:
    def remove_stopwords(self, sen):
        stop_words = stopwords.words('english')
        sentence_new = " ".join([i for i in sen if i not in stop_words])
        return sentence_new

    def generate_summary_lexRank(self, text):
        #sentences = sent_tokenize(text)
        #clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")
        #clean_sentences = [s.lower() for s in clean_sentences]
        #clean_sentences = [self.remove_stopwords(r.split()) for r in clean_sentences]
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = LexRankSummarizer()
        lexRank_summary = summarizer(parser.document, 20) 
        summary = ''

        for sentence in lexRank_summary:
            summary += ' '+str(sentence)

        return summary

# --------------- Flask Application --------------- #
app = Flask(__name__)

@app.route('/wordFreq', methods=['GET','POST'])
def wordFreq():
    word_freq = 'Word Frequency'
    if request.method == 'POST':
        url = request.form['input_url']
        general_text = general_process()
        text = general_text.convert_link_text(url)
        sum1 = word_frequency()
        summary = sum1.run_summarization(text)
        print(summary)
        return render_template('wordFreq.html', original_text = text, output_summary = summary, type_summarizer = word_freq)
    else:
        return render_template('wordFreq.html', type_summarizer = word_freq)

@app.route('/tf-idf', methods=['GET','POST'])
def tfidf():
    tf_idf = 'TF-IDF'
    if request.method == 'POST':
        url = request.form['input_url']
        general_text = general_process()
        text = general_text.convert_link_text(url)
        sum1 = TF_IDF()
        summary = sum1.run_summarization(text)
        return render_template('tf-idf.html', original_text = text, output_summary = summary, type_summarizer = tf_idf)
    else:
        return render_template('tf-idf.html', type_summarizer = tf_idf)

@app.route('/textRank', methods=['GET','POST'])
def textRank():
    textRank = 'TextRank'
    if request.method == 'POST':
        url = request.form['input_url']
        general_text = general_process()
        text = general_text.convert_link_text(url)
        sum = text_rank()
        summary = sum.generate_summary_rank(text)
        return render_template('textRank.html', original_text = text, output_summary = summary, type_summarizer = textRank)
    else:
        return render_template('textRank.html', type_summarizer = textRank)

@app.route('/lexRank', methods=['GET','POST'])
def lexRank():
    summarizer = 'LexRank'
    if request.method == 'POST':
        url = request.form['input_url']
        general_text = general_process()
        text = general_text.convert_link_text(url)
        summ = lex_rank()
        summary = summ.generate_summary_lexRank(text)
        return render_template('lexRank.html', original_text = text, output_summary = summary, type_summarizer = summarizer)
    else:
        return render_template('lexRank.html', type_summarizer = summarizer)

@app.route('/')
def home_page():
    title = 'Basic Text Summarization'
    return render_template('index.html', title=title)

if __name__ == '__main__':
    app.debug = True
    app.run()