from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

### Remove stopwords and punctuation
def remove_stop_words(x):
    stop_words = set(stopwords.words('english'))
    special_set = set(["``","''","--","'","`","'s","-","'re","'t","'m","'d","'t","An","The","A"])
    word_tokens = word_tokenize(x)
    filtered_sentence = [w for w in word_tokens if w not in stop_words
                         and w not in string.punctuation
                         and w not in special_set]
    return ' '.join([i for i in filtered_sentence if not i.isdigit()])