from os import listdir
import string
from pickle import dump
import pickle

# <----------------- Load a single document as a text -------------->
def load_doc(filename):
    file = open(filename, "rb")
    text = file.read()
    text = text.decode("utf-8", errors = "ignore")
    file.close()     
    return text

# <--- Split a document into news story and highlights(summaries) --->
def split_story(doc):
    index = doc.find('@highlight')
    story, highlights = doc[:index], doc[index:].split('@highlight')
    highlights = [h.strip() for h in highlights if len(h)>0]
    return story, highlights

# <----------------- Load all stories in a directoy ----------------->
def load_story(directory):
    all_stories = list()
    for name in listdir(directory):
        filename = directory + '/' + name
        doc = load_doc(filename)
        story, highlights = split_story(doc)
        all_stories.append({'story': story, 'highlights':highlights})
    return all_stories

# <------------- Preprocessing for story and highlights ------------>
# <-- strip source cnn office if it exists
#     normalize case to lowercase
#     remove punctuations
#     reduce the vocabulary to speed up testing models (remove numbers, remove low frequency words)
#     truncating stories to the first 5 or 10 sentences -->
def clean_lines(lines):
    cleaned = list()
    table = str.maketrans('', '', string.punctuation)
    for line in lines:
        index = line.find('(CNN) -- ')
        if index > -1:
            line = line[index+len('(CNN)'):]
        line = line.split()
        line = [word.lower() for word in line]
        line = [w.translate(table) for w in line]
        line = [word for word in line if word .isalpha()]
        cleaned.append(' '.join(line))
    cleaned = [c for c in cleaned if len(c) > 0]
    return cleaned


directory = 'cnn/stories/'
stories = load_story(directory)
print('Loaded Stories %d' % len(stories))

for example in stories:
	example['story'] = clean_lines(example['story'].split('\n'))
	example['highlights'] = clean_lines(example['highlights'])


dump(stories, open('cnn_dataset.pkl', 'wb'))
# from nltk.tokenize import word_tokenize, sent_tokenize

# stories = pickle.load(open('cnn_dataset.pkl', 'rb'))
# original_text = stories[6]['story']
# print(stories[6]['highlights'])
# summary = ''
# for text in original_text:
#     summary += ' '+text+'.'
# sentences = sent_tokenize(summary)
# print(sentences)