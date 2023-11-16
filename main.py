from optparse import OptionParser
import time
import re

import pandas as pd
import numpy as np

from bert_preprocess import text_preprocessing , bert_preprocessing
from trainer import initialize_model, set_seed
from trainer import train

from transformers import AutoTokenizer

from torch.utils.data import TensorDataset , DataLoader , RandomSampler , SequentialSampler , ConcatDataset
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report



def get_args_parser():
    parser = OptionParser()

    parser.add_option("--model", dest="model_name", help="bert-base-cased | ProsusAI/finbert | bert-large-uncased", default="bert-base-cased", type="string")
    parser.add_option("--num_class", dest="num_class", help="number of the classes in the output layer",
                      default=2, type="int")
    parser.add_option("--data_dir", dest="data_dir", help="dataset dir",default="./dataset/amazon_news_daily_summarized_spacy.csv", type="string")
    parser.add_option("--column", dest="column", help="column to consider (title | text)",
                      default="text", type="string")
    parser.add_option("--mode", dest="mode", help="train | test", default="train", type="string")
    parser.add_option("--batch_size", dest="batch_size", help="batch size", default=16, type="int")
    parser.add_option("--num_epoch", dest="num_epoch", help="num of epoches", default=30, type="int")
    parser.add_option("--optimizer", dest="optimizer", help="AdamW", default="AdamW", type="string")
    parser.add_option("--lr", dest="lr", help="learning rate", default=1e-5, type="float")
    parser.add_option("--token_len" , dest = "tkl" , help="tokenization length (512 , base) | (1024 , large) | (512 , finbert)" , default="512" , type="int")
    parser.add_option("--train-test-split", dest="split_ratio", help="split size", default=0.25, type="float")
    parser.add_option("--k-fold-cross-val", dest="cross_val", help="K fold cross validation", default=True, action = 'store_true')
    parser.add_option("--summarize", dest="summary", help="Summarize the text", default = False, action = 'store_true')
    parser.add_option("--summary_model", dest="summary_type", help="summary model (spacy/transformers)", default = "spacy", type="string")
    parser.add_option("--folds", dest="kfolds", help="K folds", default=5, type="int")
    (options, args) = parser.parse_args()

    return options

def evaluate(model , device , val_dataloader):
    """
    Evaluate performance on the validation set
    """
    model.eval()

    val_preds = []
    val_true = []

    for batch in val_dataloader:

        val_input_ids , val_attn_mask , val_labels = tuple( x.to(device) for x in batch)
        val_labels = val_labels.type(torch.LongTensor).to(device)
        
        with torch.no_grad():
            val_logits = model(val_input_ids , val_attn_mask)

        preds = torch.argmax(val_logits , dim = 1).flatten()
        val_preds += preds.cpu().numpy().tolist()
        val_true += val_labels.cpu().numpy().tolist()
        
    return val_preds , val_true


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
        pre_processed_data.to_csv(f"./dataset/{file_name}_summarized_{args.summary_type}.csv" , index = False)
        end_time = time.time()
        print(f"Total time taken : {end_time - start_time}")
        print("Successfully pre-processed the data")

    else:
        k_folds = args.kfolds

        pre_processed_data = pd.read_csv(args.data_dir)
    
        # If the path to the train and val tokenized data exists
        # if os.path.exists(f"./tokenized_data/{file_name}_{model_name}_{args.summary_type}_{args.column}_{args.tkl}_{args.split_ratio}"):
            
        #     start_time = time.time()

        #     train_inputs = torch.load(f"./tokenized_data/{file_name}_{model_name}_{args.summary_type}_{args.column}_{args.tkl}_{args.split_ratio}/train_input.pt")
        #     train_masks = torch.load(f"./tokenized_data/{file_name}_{model_name}_{args.summary_type}_{args.column}_{args.tkl}_{args.split_ratio}/train_mask.pt")
        #     train_labels = torch.load(f"./tokenized_data/{file_name}_{model_name}_{args.summary_type}_{args.column}_{args.tkl}_{args.split_ratio}/train_labels.pt")

        #     val_inputs = torch.load(f"./tokenized_data/{file_name}_{model_name}_{args.summary_type}_{args.column}_{args.tkl}_{args.split_ratio}/val_input.pt")
        #     val_masks = torch.load(f"./tokenized_data/{file_name}_{model_name}_{args.summary_type}_{args.column}_{args.tkl}_{args.split_ratio}/val_mask.pt")
        #     val_labels = torch.load(f"./tokenized_data/{file_name}_{model_name}_{args.summary_type}_{args.column}_{args.tkl}_{args.split_ratio}/val_labels.pt")
            
        #     end_time = time.time()
        #     print(f"Total time taken : {end_time - start_time}")
        #     print("Successfully Loaded the Input variables for the model from existing directory")
            
        # else:
        start_time = time.time()

        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        
        # start_time = time.time()

        X_train , X_val , y_train , y_val = train_test_split(pre_processed_data[args.column].apply(str).values , pre_processed_data.label.values , test_size = args.split_ratio , random_state = 2020)
        train_labels = torch.tensor(y_train)
        val_labels = torch.tensor(y_val)
        
        # end_time = time.time()
        # print(f"Total time taken : {end_time - start_time}")
        # print("Successfully made train and test splits")

        start_time = time.time()

        # os.makedirs(f"./tokenized_data/{file_name}_{model_name}_{args.summary_type}_{args.column}_{args.tkl}_{args.split_ratio}")
        
        # Tokenize the sentence and truncate it to max length of the model
        train_inputs  , train_masks = bert_preprocessing(tokenizer , X_train , args.tkl)
        # torch.save(train_inputs , f"./tokenized_data/{file_name}_{model_name}_{args.summary_type}_{args.column}_{args.tkl}_{args.split_ratio}/train_input.pt")
        # torch.save(train_masks , f"./tokenized_data/{file_name}_{model_name}_{args.summary_type}_{args.column}_{args.tkl}_{args.split_ratio}/train_mask.pt")
        # torch.save(train_labels , f"./tokenized_data/{file_name}_{model_name}_{args.summary_type}_{args.column}_{args.tkl}_{args.split_ratio}/train_labels.pt")

        end_time = time.time()
        print(f"Total time taken : {end_time - start_time}")
        print("Successfully made train inputs and masks")

        start_time = time.time()

        val_inputs  , val_masks = bert_preprocessing(tokenizer , X_val , args.tkl)
        # torch.save(val_inputs , f"./tokenized_data/{file_name}_{model_name}_{args.summary_type}_{args.column}_{args.tkl}_{args.split_ratio}/val_input.pt")
        # torch.save(val_masks , f"./tokenized_data/{file_name}_{model_name}_{args.summary_type}_{args.column}_{args.tkl}_{args.split_ratio}/val_mask.pt")
        # torch.save(val_labels , f"./tokenized_data/{file_name}_{model_name}_{args.summary_type}_{args.column}_{args.tkl}_{args.split_ratio}/val_labels.pt")

        end_time = time.time()
        print(f"Total time taken : {end_time - start_time}")
        print("Successfully made val inputs and masks")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch_size = args.batch_size

        start_time = time.time()

        # Create the DataLoader for our training set
        train_data = TensorDataset(train_inputs , train_masks , train_labels)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data , batch_size = batch_size , sampler= train_sampler)

        end_time = time.time()
        print(f"Total time taken : {end_time - start_time}")
        print("Dataloader for training set")

        start_time = time.time()

        # Create the DataLoader for our validation set
        val_data = TensorDataset(val_inputs, val_masks, val_labels)
        val_sampler = SequentialSampler(val_data)
        val_dataloader = DataLoader(val_data, batch_size=batch_size , sampler = val_sampler)

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

        # loss function definition
        loss_fn = nn.CrossEntropyLoss(weight = weights).to(device)

        if args.cross_val:
            dataset = ConcatDataset([train_data, val_data])
            kfold = KFold(n_splits=k_folds, shuffle=True)
            classification_reports = {}
            for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
    
                # Print
                print(f'FOLD {fold + 1}')
                print('--------------------------------')
                
                # Sample elements randomly from a given list of ids, no replacement.
                train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
                val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)
                
                # Define data loaders for training and testing data in this fold
                train_dataloader = DataLoader(
                                dataset, 
                                batch_size=batch_size, sampler=train_subsampler)
                val_dataloader = DataLoader(
                                dataset,
                                batch_size=batch_size, sampler=val_subsampler)
                
                # Init the network
                bert_classifier, optimizer, scheduler , lr_scheduler = initialize_model(train_dataloader , args.model_name , epochs = args.num_epoch , num_classes= args.num_class , device = device , lr = args.lr )
                set_seed()
                train(args , file_name , model_name , bert_classifier, loss_fn , train_dataloader, device , optimizer , scheduler, lr_scheduler , fold + 1, val_dataloader , epochs=args.num_epoch, evaluation=True)
                
                best_model = torch.load(f"./models/{file_name}_{model_name}_{args.summary_type}_{args.column}_{args.tkl}_{args.batch_size}_{args.lr}/{fold + 1}.pt")
                bert_classifier, optimizer, scheduler , lr_scheduler = initialize_model(train_dataloader , args.model_name , epochs = args.num_epoch , num_classes= args.num_class , device = device , lr = args.lr )
                bert_classifier.load_state_dict(best_model['model_state_dict'])
                val_preds , val_true = evaluate(bert_classifier , device , val_dataloader)
                # print(len(val_preds) , len(val_true))
                # print(val_preds[:5] , val_true[:5])
                report = classification_report(val_true , val_preds , output_dict=True)
                classification_reports[f"report_{fold + 1}"] = report
            torch.save(classification_reports , f"./plots/{file_name}_{model_name}_{args.summary_type}_{args.column}_{args.tkl}_{args.batch_size}_{args.lr}_report.pt")
        else:
            bert_classifier, optimizer, scheduler , lr_scheduler = initialize_model(train_dataloader , args.model_name , epochs = args.num_epoch , num_classes= args.num_class , device = device , lr = args.lr )
            set_seed()
            train(args , file_name , model_name , bert_classifier, loss_fn , train_dataloader, device , optimizer , scheduler, lr_scheduler , 0, val_dataloader , epochs=args.num_epoch, evaluation=True)

if __name__ == '__main__':
    args = get_args_parser()
    main(args)