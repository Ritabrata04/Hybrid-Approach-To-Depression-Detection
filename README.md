# Deep Learning for Depression Detection

This repository is dedicated to an investigation of various Deep Learning techniques for depression detection in text data. The techniques explored include several variants of Recurrent Neural Networks (RNNs) such as Long Short-Term Memory (LSTM), Gated Recurrent Unit (GRU), Bidirectional LSTM (BiLSTM), Transformer model using BERT (Bidirectional Encoder Representations from Transformers), and a simple Feed-Forward Neural Network (FFNN) used as a baseline model. 

## Dataset

The dataset used is a collection of text data gathered from different threads of depression-related discussions. Each sample has been labeled as either "depression" or "non-depression" based on the content of the discussion. The text has been preprocessed to be in a suitable format for analysis by various machine learning algorithms.

## Models

The models used in the study include:

1. **GRU:** This model utilizes a Gated Recurrent Unit architecture, a variant of RNNs, which is capable of preserving long-term dependencies in sequence data while being computationally more efficient than traditional LSTMs.

2. **LSTM:** The Long Short-Term Memory model is another variant of RNNs. It effectively captures long-term dependencies in sequence data by maintaining a separate memory cell that updates and exposes its content only when deemed necessary.

3. **BiLSTM:** The Bidirectional LSTM model is an extension of traditional LSTM. It processes the data in both forward and backward directions to keep the context from both past and future data.

4. **BERT:** BERT is a Transformer model specifically designed for NLP tasks. It uses bidirectional training of transformers, which allows for a deep understanding of the context of a word based on all its surroundings (left and right of the word).

5. **FFNN (Baseline):** This is a simple Feed-Forward Neural Network used as a baseline model for the comparison of the performance of other complex models.

## Autoencoder

An Autoencoder was implemented to understand the semantics of the text data, which was then used for unsupervised learning using KMeans clustering.

## Results

The models' performance was evaluated based on precision, recall, F1-score, and accuracy. The results are as follows:

| Model  | Precision | Recall | F1-Score | Accuracy |
| ------ | --------- | ------ | -------- | -------- |
| GRU    | 0.51      | 0.51   | 0.51     | 0.51     |
| LSTM   | 0.95      | 0.95   | 0.95     | 0.95     |
| BERT   | 0.96      | 0.96   | 0.96     | 0.96     |
| FFNN   | 0.99      | 0.99   | 0.99     | 0.99     |
| BiLSTM | 0.96      | 0.96   | 0.96     | 0.95     |

Graphical and tabular comparisons between these models are available in the 'RESULTS.pdf' file.

## Repository Files

1. **Baseline.py:** Contains the code for the implementation of the FFNN model.
2. **BERT.py:** Contains the code for the implementation of the BERT model.
3. **DATA VISUALISATION.py:** Contains the code for various data visualization techniques implemented on the dataset.
4. **DEPRESSION_DETECTION_USING_LSTM_AND_GRU.py:** Contains the code for the implementation of both the LSTM and GRU models.
5. **RESULTS.pdf:** Contains graphical and tabular comparisons between the implemented models.
6. **wordcloud_depression.png:** A wordcloud visualization showing the most common words in depression-related sentence threads.
   
Please note that you might need to install specific Python libraries to be able to run these scripts, and be sure to adjust the path of the data according to your local setup.
