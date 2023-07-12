# Depression Detection Using LSTM and GRU

This repository houses a project focused on detecting signs of depression using text data. The project uses two powerful Recurrent Neural Network (RNN) architectures, Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU), both highly effective for sequential data like text. 

## Project Overview

This project employs natural language processing (NLP) and deep learning techniques to build a model that can potentially identify depressive indications in text data. It leverages LSTM and GRU, two types of RNNs, which are commonly used in tasks involving sequential data.

**Long Short-Term Memory (LSTM)**: LSTM networks are a type of RNN that use special units in addition to standard units. LSTM units include a 'memory cell' that can maintain information in memory for long periods of time. This mitigates the vanishing gradient problem encountered with traditional RNNs and makes LSTM networks ideal for learning from important experiences that have very long time lags.

**Gated Recurrent Unit (GRU)**: The GRU is the newer generation of RNN and is pretty similar to an LSTM. GRU's got rid of the cell state and used the hidden state to transfer information. It also only has two gates, a reset gate and update gate, making it simpler and often more efficient computationally.

## Repository Contents

- `DEPRESSION_DETECTION_USING_LSTM.py`: The Python script contains all the steps involved in the project from data cleaning, tokenization, and sequencing to training and evaluation of LSTM and GRU models.

- `LICENSE`: The license file states the terms under which this project is distributed.

- `dataset distribution.png`: A visual overview of the distribution of the dataset.

- `depression_dataset_reddit_cleaned.csv.zip`: This compressed file contains a preprocessed dataset obtained from Reddit discussions related to depression.

- `gru confusion matrix.png`: A confusion matrix of the GRU model's performance, visualizing the true positives, true negatives, false positives, and false negatives.

- `lstm confusion matrix.png`: A confusion matrix of the LSTM model's performance.

## How to Use

1. Clone this repository to your local machine.
2. Ensure that you have the necessary Python libraries installed (as mentioned in the Python file).
3. Unzip `depression_dataset_reddit_cleaned.csv.zip` to get the dataset.
4. Execute `DEPRESSION_DETECTION_USING_LSTM.py` to replicate the project.

## Requirements

- Python 3.7+
- Libraries: Pandas, NumPy, Matplotlib, Seaborn, TensorFlow, scikit-learn, NLTK

**Note**: While this project aims to build a model for depression detection based on text data, it should not serve as a substitute for professional diagnosis. For advice and treatment, always consult with a licensed healthcare professional.

## Contributions

Contributions, issues, and feature requests are welcome. For major changes, please open an issue first to discuss your proposals.

## License

This project is distributed under the MIT License - see the [LICENSE] file for more details.

## Contact

 Ritabrata Chakraborty-ritabrata04@gmail.com

Project Link: https://github.com/Ritabrata04/Depression-Detection-Using-LSTM-And-GRU

Feel free to reach out if you have any questions or need further clarification about the project.
