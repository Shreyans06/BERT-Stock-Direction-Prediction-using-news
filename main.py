import pandas as pd
import re
from bert_preprocess import text_preprocessing , bert_preprocessing
from transformers import BertTokenizer , BertModel , RobertaTokenizer
import numpy as np
from transformers import AutoTokenizer
from torch.utils.data import TensorDataset , DataLoader , RandomSampler , SequentialSampler
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
from model import BertClassifier
from trainer import initialize_model, set_seed
from trainer import train
from sklearn.model_selection import train_test_split
import os
from optparse import OptionParser
import time
from summary import summarize

def get_args_parser():
    parser = OptionParser()

    parser.add_option("--model", dest="model_name", help="bert-base-uncased | ProsusAI/finbert | bert-large-uncased", default="bert-base-uncased", type="string")
    parser.add_option("--num_class", dest="num_class", help="number of the classes in the output layer",
                      default=2, type="int")
    parser.add_option("--data_dir", dest="data_dir", help="dataset dir",default=None, type="string")
    parser.add_option("--column", dest="column", help="column to consider (title | text)",
                      default="text", type="string")
    parser.add_option("--mode", dest="mode", help="train | test", default="train", type="string")
    parser.add_option("--batch_size", dest="batch_size", help="batch size", default=16, type="int")
    parser.add_option("--num_epoch", dest="num_epoch", help="num of epoches", default=30, type="int")
    parser.add_option("--optimizer", dest="optimizer", help="AdamW", default="AdamW", type="string")
    parser.add_option("--lr", dest="lr", help="learning rate", default=5e-5, type="float")
    parser.add_option("--token_len" , dest = "tkl" , help="tokenization length (512 , base) | (1024 , large) | (512 , finbert)" , default="512" , type="int")
    parser.add_option("--train-test-split", dest="split_ratio", help="split size", default=0.1, type="float")
    parser.add_option("--k-fold-cross-val", dest="cross_val", help="K fold cross validation", default=True, action = 'store_true')
    parser.add_option("--summarize", dest="summary", help="Summarize the text", default = False, action = 'store_true')
    parser.add_option("--summary_model", dest="summary_type", help="summary model (spacy/transformers)", default = "spacy", type="string")

    (options, args) = parser.parse_args()

    return options

def main(args):

    print(args) 
    print("-" * 70)
    
    assert args.data_dir is not None
    assert args.column is not None

    torch.cuda.empty_cache()
    
    # get the input data file name
    file_location = args.data_dir.split(".csv")
    file_name = file_location[0].split("/")[-1] 

    # get the pre-trained model name
    delimiters = "-" , "/"
    regex_pattern = '|'.join(delimiters)
    model_name = "_".join(re.split(regex_pattern , args.model_name))


    if args.summary:
        start_time = time.time()
        # Reading the input data
        input_data = pd.read_csv(args.data_dir)
        input_data = input_data[~input_data['labels'].isna()]
        input_data['label'] = input_data['labels'].astype('int32')

        # Keeping only required columns
        data = input_data[[args.column , 'label']]

        # Making a copy and applying basic text pre-processing
        pre_processed_data = data.copy()
        pre_processed_data.reset_index(drop = True)
        pre_processed_data = pre_processed_data.dropna()
        pre_processed_data[args.column] = pre_processed_data[args.column].apply(str).apply(lambda x: text_preprocessing(x , args.summary_type))
        pre_processed_data = pre_processed_data.dropna()
        pre_processed_data.to_csv(f"/home/snola136/SWM/dataset/{file_name}_summarized_{args.summary_type}.csv" , index = False)
        end_time = time.time()
        print(f"Total time taken : {end_time - start_time}")
        print("Successfully pre-processed the data")

    else:
        pre_processed_data = pd.read_csv(args.data_dir)
    
        # If the path to the train and val tokenized data exists
        if os.path.exists(f"./tokenized_data/{file_name}_{model_name}_{args.summary_type}_{args.column}_{args.tkl}_{args.split_ratio}"):
            
            start_time = time.time()

            train_inputs = torch.load(f"./tokenized_data/{file_name}_{model_name}_{args.summary_type}_{args.column}_{args.tkl}_{args.split_ratio}/train_input.pt")
            train_masks = torch.load(f"./tokenized_data/{file_name}_{model_name}_{args.summary_type}_{args.column}_{args.tkl}_{args.split_ratio}/train_mask.pt")
            train_labels = torch.load(f"./tokenized_data/{file_name}_{model_name}_{args.summary_type}_{args.column}_{args.tkl}_{args.split_ratio}/train_labels.pt")

            val_inputs = torch.load(f"./tokenized_data/{file_name}_{model_name}_{args.summary_type}_{args.column}_{args.tkl}_{args.split_ratio}/val_input.pt")
            val_masks = torch.load(f"./tokenized_data/{file_name}_{model_name}_{args.summary_type}_{args.column}_{args.tkl}_{args.split_ratio}/val_mask.pt")
            val_labels = torch.load(f"./tokenized_data/{file_name}_{model_name}_{args.summary_type}_{args.column}_{args.tkl}_{args.split_ratio}/val_labels.pt")
            
            end_time = time.time()
            print(f"Total time taken : {end_time - start_time}")
            print("Successfully Loaded the Input variables for the model from existing directory")
            
        else:
            start_time = time.time()

            tokenizer = AutoTokenizer.from_pretrained(args.model_name)
            
            start_time = time.time()

            if args.cross_val:
                X_train , X_val , y_train , y_val = train_test_split(pre_processed_data[args.column].apply(str).values , pre_processed_data.label.values , test_size = args.split_ratio , random_state = 2020)
                train_labels = torch.tensor(y_train)
                val_labels = torch.tensor(y_val)
            
            end_time = time.time()
            print(f"Total time taken : {end_time - start_time}")
            print("Successfully made train and test splits")

            start_time = time.time()

            os.makedirs(f"./tokenized_data/{file_name}_{model_name}_{args.summary_type}_{args.column}_{args.tkl}_{args.split_ratio}")
            
            # Tokenize the sentence and truncate it to max length of the model
            train_inputs  , train_masks = bert_preprocessing(tokenizer , X_train , args.tkl)
            torch.save(train_inputs , f"./tokenized_data/{file_name}_{model_name}_{args.summary_type}_{args.column}_{args.tkl}_{args.split_ratio}/train_input.pt")
            torch.save(train_masks , f"./tokenized_data/{file_name}_{model_name}_{args.summary_type}_{args.column}_{args.tkl}_{args.split_ratio}/train_mask.pt")
            torch.save(train_labels , f"./tokenized_data/{file_name}_{model_name}_{args.summary_type}_{args.column}_{args.tkl}_{args.split_ratio}/train_labels.pt")

            end_time = time.time()
            print(f"Total time taken : {end_time - start_time}")
            print("Successfully made train inputs and masks")

            start_time = time.time()

            val_inputs  , val_masks = bert_preprocessing(tokenizer , X_val , args.tkl)
            torch.save(val_inputs , f"./tokenized_data/{file_name}_{model_name}_{args.summary_type}_{args.column}_{args.tkl}_{args.split_ratio}/val_input.pt")
            torch.save(val_masks , f"./tokenized_data/{file_name}_{model_name}_{args.summary_type}_{args.column}_{args.tkl}_{args.split_ratio}/val_mask.pt")
            torch.save(val_labels , f"./tokenized_data/{file_name}_{model_name}_{args.summary_type}_{args.column}_{args.tkl}_{args.split_ratio}/val_labels.pt")

            end_time = time.time()
            print(f"Total time taken : {end_time - start_time}")
            print("Successfully made val inputs and masks")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch_size = args.batch_size

        start_time = time.time()

        # Create the DataLoader for our training set
        train_data = TensorDataset(train_inputs , train_masks , train_labels)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data , batch_size = batch_size , sampler = train_sampler)

        end_time = time.time()
        print(f"Total time taken : {end_time - start_time}")
        print("Dataloader for training set")

        start_time = time.time()

        # Create the DataLoader for our validation set
        val_data = TensorDataset(val_inputs, val_masks, val_labels)
        val_sampler = SequentialSampler(val_data)
        val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

        end_time = time.time()
        print(f"Total time taken : {end_time - start_time}")
        print("Dataloader for val set")

        # compute the class weights
        class_weights = compute_class_weight(class_weight = "balanced", classes = np.unique(pre_processed_data['label'].tolist()), y = train_labels.tolist() )
        print("class weights are {} for {}".format(class_weights,np.unique(train_labels)))

        # wrap class weights in tensor
        weights = torch.tensor(class_weights,dtype=torch.float)

        # push weights to GPU
        weights = weights.to(device)

        loss_fn = nn.CrossEntropyLoss(weight = weights).to(device)
        loss_fn = nn.CrossEntropyLoss().to(device)

        start_time = time.time()

        bert_classifier, optimizer, scheduler , lr_scheduler = initialize_model(train_dataloader , args.model_name , epochs = args.num_epoch , num_classes= args.num_class , device = device , lr = args.lr)
        end_time = time.time()
        print(f"Total time taken : {end_time - start_time}")
        print("Initialized the model")
        set_seed()
        train(args , file_name , model_name , bert_classifier, loss_fn , train_dataloader, device , optimizer , scheduler, lr_scheduler , val_dataloader , epochs=args.num_epoch, evaluation=True)

if __name__ == '__main__':
    args = get_args_parser()
    main(args)