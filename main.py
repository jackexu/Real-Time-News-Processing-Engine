import sys, os, warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from Webscraping import get_text_from_url
from Embedding import text_embedding
from Prediction import generate_label
from Cleaning import remove_stop_words
import time

embed_size = 300 # how big is each word vector
max_features = 20000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 100 #
model_final = 'DL-model_300_all.h5'

url = 'https://www.cnn.com/2019/11/30/politics/donald-trump-impeachment-inquiry-strategy/index.html'

#%%
start = time.time()
test_text = get_text_from_url(url)
end = time.time()
print(test_text)
print('- Cost {}'.format(end-start))

# #%%
# start = time.time()
# test_text = remove_stop_words(test_text)
# end = time.time()
# print(test_text)
# print('- Cost {}'.format(end-start))

#%%
start = time.time()
x_test = text_embedding(test_text, max_features, maxlen)
end = time.time()
print(x_test)
print('- Cost {}'.format(end-start))

#%%
start = time.time()
label_pred, label_df = generate_label(model_final, x_test)
end = time.time()
print(label_pred)
print(label_df)
print('- Cost {}'.format(end-start))