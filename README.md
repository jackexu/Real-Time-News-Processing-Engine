# Real-Time-News-Processing-Engine

A capstone project from the M.Sc. Analytics program at the University of Chicago

## Abstract
The purpose of this project is to perform Natural Language Processing (NLP) on news reports and provide real-time classification. We aim to extract insights from unstructured text to evaluate the importance and relevance of each report pertaining to social and political activities. We collected and analyzed publicly available more than 30K articles. We applied word and document embeddings along with a robust self-supervised learning model to perform accurate document classification. Our model achieved around 85% accuracy on the class prediction and allow readers to be informed about major themes of incoming news in real-time.

This page only contains the Flask UI part.

## Flask UI

This Flask app is an UI that allows user to input an URL and present the label prediction and corresponding probability.

### Initial Page
![Flask_UI](Flask_UI.png)

### Model Result
![Flask_UI](Flask_UI_Result.png)

## Latest version bug fixs

Solved model loading issue under tensorflow 2.0 and Keras 2.3 by using 
```
from tensorflow.kerea.models import load_model
```
