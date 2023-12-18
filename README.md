# BERT-Stock-Direction-Prediction-using-news


### This project focuses on harnessing the power of viral tweets and news data to accurately predict stock prices for two prominent technology companies, Amazon and Apple. We employ natural language processing (NLP) techniques and machine learning algorithms to extract insights from social media and news sources. The project presents the methodology and techniques used to achieve this goal and evaluate the predictive performance of our models — a stock price direction prediction project built using the BERT pre-trained model.

### CODE
1. ``` trainer.py ``` contains the train helper function for the BERT.
2. ``` summary.py ``` contains the code to summarize the text.
3. ``` plot.py ``` contains code to get plots for BERT
4. ``` model.py ``` contains the code for defining the model architecture
5. ``` inference.py ``` contains the code for running the inference engine for BERT
6. ``` main.py ``` contains the base code for BERT
7. ``` bert_preprocess.py ``` contains code to pre-process the text for BERT

Steps to run the project: 
1. Open the file main.py
2. To run the main.py file use the command ``` python main.py --data_dir <dataset-dir> --model <model_name> --token_len <token_length> --folds <K_fold>```
