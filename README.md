# sentiment_analysis
This project implemented neural network and machine learning to classify tweets with sentiment labels.

## BERT

Please run the following code in the BERT_Three_Point.ipynb notebook
The dataset is Tweets_airlines.csv

Training Command: 
- data = read_airline_data("Tweets_airlines.csv")
- split_data(data)
- dataset_train, dataset_val, dataset_test = prepare_dataset()
- train(dataset_train, dataset_val)

Testing Command:
- test(model_path = 'second_finetuned_BERT_epoch_5.model')

## LSTM Three Point Scale

Please run the following code in the RNN_Three_Point.ipynb notebook
The dataset is Tweets_airlines.csv

Training Command:
- data = read_airline_data(data_path = "Tweets_airlines.csv")
- split_data(data)
- model = train("airline_pkl/train.pkl")
- torch.save(model.state_dict(), "RNN_airline_model_no_exclamation_with_vader")

Testing Command:
- test("RNN_airline_model_no_exclamation_with_vader", "airline_pkl/test.pkl")

## LSTM Five Point Scale

Please run the following code in the Five_Point.ipynb notebook
The dataset is Twitter-sentiment-self-drive-DFE.csv

Training Command:

- data = read_drive_data("Twitter-sentiment-self-drive-DFE.csv")
- split_data(data)
- model = train("drive_pkl/train_drive.pkl")
- torch.save(model.state_dict(), "RNN_drive_model_no_exclamation_no_vader")

Testing Command:
- test("RNN_drive_model_no_exclamation_no_vader", "drive_pkl/test_drive.pkl")

## Machine Learning Five Point Scale

Please go into the ipynb file and run the cell for the all the six machine learning models.
