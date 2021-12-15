# sentiment_analysis
This project implemented neural network and machine learning to classify tweets with sentiment labels.

## RNN models used Glove Embeddings, and the file is not provided in this repository. 

## BERT

Please run the following code in the BERT_Three_Point.ipynb notebook <br>
The dataset is Tweets_airlines.csv

Training Command: 
- data = read_airline_data("Tweets_airlines.csv")
- split_data(data)
- dataset_train, dataset_val, dataset_test = prepare_dataset()
- train(dataset_train, dataset_val)

Testing Command:
- test(model_path = 'second_finetuned_BERT_epoch_5.model')

## LSTM Three Point Scale

Please run the following code in the RNN_Three_Point.ipynb notebook <br>
The dataset is Tweets_airlines.csv

These are the hyperparameters used
They are also provided in the notebook
- EMBEDDING_DIM = 50
- HIDDEN_DIM = 16
- NUM_LAYERS = 1
- MAX_SEQ_LENGTH = 20
- LEARNING_RATE = 0.005
- EPOCH = 3
- BATCH_SIZE = 10
- WEIGHT_DECAY = 0
- GLOVE_PATH = "glove.6B 2"

Training Command:
- data = read_airline_data(data_path = "Tweets_airlines.csv")
- split_data(data)
- model = train("airline_pkl/train.pkl")
- torch.save(model.state_dict(), "RNN_airline_model_no_exclamation_with_vader")

Testing Command:
- test("RNN_airline_model_no_exclamation_with_vader", "airline_pkl/test.pkl")

## LSTM Five Point Scale

Please run the following code in the Five_Points.ipynb notebook <br>
The dataset is Twitter-sentiment-self-drive-DFE.csv

These are the hyperparameters used
They are also provided in the notebook
- EMBEDDING_DIM = 50
- HIDDEN_DIM = 16
- NUM_LAYERS = 1
- MAX_SEQ_LENGTH = 20
- LEARNING_RATE = 0.005
- EPOCH = 3
- BATCH_SIZE = 10
- WEIGHT_DECAY = 0
- GLOVE_PATH = "glove.6B 2"

Training Command:

- data = read_drive_data("Twitter-sentiment-self-drive-DFE.csv")
- split_data(data)
- model = train("drive_pkl/train_drive.pkl")
- torch.save(model.state_dict(), "RNN_drive_model_no_exclamation_no_vader")

Testing Command:
- test("RNN_drive_model_no_exclamation_no_vader", "drive_pkl/test_drive.pkl")

## Machine Learning Five Point Scale

Please run the following code to prepare data for machine learning models in Five_Points.ipynb. <br>
The code is also provided in the notebook. <br>
Then, go into the ipynb file and run the cell for the all the six machine learning models <br>
The dataset is Twitter-sentiment-self-drive-DFE.csv

- data_ml = read_drive_data("Twitter-sentiment-self-drive-DFE.csv")
- split_data(data_ml)

- train_file = open("drive_pkl/train_drive.pkl", "rb")
- data_ml = pickle.load(train_file)
- train_file.close()

- train_tfidf_matrix = extract_feature(data_ml)
- data_tfidf = pd.DataFrame(train_tfidf_matrix.todense())
- data_vader = pd.concat([data_tfidf, data_ml["vader_sentiment"].reset_index(drop=True)], axis = 1)
- x_train_tfidf, x_valid_tfidf, y_train, y_valid = train_test_split(data_tfidf,data_ml['sentiment'],test_size=0.1,random_state=17)
