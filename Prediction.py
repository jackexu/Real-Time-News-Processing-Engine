import numpy as np
import pandas as pd
"""
"from tensorflow.keras.models import load_model" is for tensorflow 2.0 and Keras 2.3.1
"from keras.models import load_model" is for tensorflow 1.15.0 and Keras 2.2.5
"""
from tensorflow.keras.models import load_model
from keras import backend as K

# fix random seed for reproducibility
np.random.seed(2019)

def generate_label_df(y_test_hat):
    df = pd.DataFrame({'Law': y_test_hat[:, 0], 'Pharma': y_test_hat[:, 1],
                    'Politics':y_test_hat[:, 2], 'Protest':y_test_hat[:, 3],
                    'Threat':y_test_hat[:, 4]})
    df['Highest_Prob'] = df.max(axis=1)
    df['Pred_Label'] = df.idxmax(axis=1)
    return df

def generate_label(model_file, X_te):
    # Load model
    model = load_model(model_file)
    # Make prediction
    y_test_hat = model.predict(X_te, batch_size=64, verbose=1)
    # Add clear_session after prediction
    K.clear_session()  # So that the program can be run multi times without restart
    # Generate label for the prediction
    label_df = generate_label_df(y_test_hat)
    # Convert to string
    label_pred = label_df.copy().Pred_Label.to_string(index=False)

    return label_pred, label_df