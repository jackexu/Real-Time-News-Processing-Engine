import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def text_embedding(input_text, max_features, maxlen):
    input_text_list = [input_text]
    df_test = pd.DataFrame(input_text_list, columns=['orig_text'])

    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(df_test.orig_text)
    sequences = tokenizer.texts_to_sequences(df_test.orig_text)
    X_test = pad_sequences(sequences, maxlen=maxlen)
    return X_test