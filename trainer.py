from model import BertClassifier , finBertClassifier , DistillBERTClass
import torch
from transformers import get_linear_schedule_with_warmup
from  torch.optim import AdamW , Adam
import numpy as np
import random

class ValidationLossEarlyStopping:
    def __init__(self, patience=1, min_delta=0.0):
        self.patience = patience  # number of times to allow for no improvement before stopping the execution
        self.min_delta = min_delta  # the minimum change to be counted as improvement
        self.counter = 0  # count the number of times the validation accuracy not improving
        self.min_validation_loss = np.inf

    # return True when validation loss is not decreased by the `min_delta` for `patience` times 
    def early_stop_check(self, validation_loss):
        if ((validation_loss+self.min_delta) < self.min_validation_loss):
            self.min_validation_loss = validation_loss
            self.counter = 0  # reset the counter if validation loss decreased at least by min_delta
        elif ((validation_loss+self.min_delta) > self.min_validation_loss):
            self.counter += 1 # increase the counter if validation loss is not decreased by the min_delta
            if self.counter >= self.patience:
                return True
        return False
    

def initialize_model(dataloader , model_name , device = 'cpu' , epochs = 4 , num_classes = 2 , lr = 2e-5):
    """
    Initialize the Bert Classifier , and other logistics for training
    """
    if model_name == "bert-base-uncased":
        bert_classifier = BertClassifier(model_name , num_classes= num_classes , freeze_bert = False)
    elif model_name == "distilbert-base-uncased":
        bert_classifier = DistillBERTClass(num_classes= num_classes)
    else:
        bert_classifier = finBertClassifier(model_name , num_classes= num_classes )
    
    bert_classifier.to(device)
    
    if device == 'cuda':
        bert_classifier = torch.nn.DataParallel(bert_classifier)

    optimizer = AdamW(
        bert_classifier.parameters(),
        lr = lr,
        eps = 1e-8
    )

    total_steps = len(dataloader) * epochs

    scheduler = get_linear_schedule_with_warmup(optimizer , num_warmup_steps = 0 , num_training_steps = total_steps)

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5, verbose=True)

    return bert_classifier , optimizer , scheduler , lr_scheduler

def set_seed(seed_value=42):
    """Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

def train(args , file_name , model_name , model , loss_fn , train_dataloader , device , optimizer , scheduler , lr_scheduler , val_dataloader = None , epochs = 4 , evaluation=True):
    """
    Train a BertClassifier model
    """
    print("Start Training..\n")

    patience = 10
    early_stopping = ValidationLossEarlyStopping(patience=patience, min_delta= 10)

    train_loss_epochs = []
    val_loss_epochs = []
    val_accuracy_epochs = []
    best_val_loss = float('inf')

    for epoch in range(epochs):
        print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} ")
        
        print("-"*70)

        total_loss , batch_loss , batch_count = 0 , 0 , 0

        model.train()

        for step , batch in enumerate(train_dataloader):

            batch_count += 1
            
            input_ids , attn_mask , labels = tuple(x.to(device) for x in batch)
            
            labels = labels.type(torch.LongTensor).to(device)
              
            model.zero_grad()

            logits = model(input_ids , attn_mask)

            loss = loss_fn(logits , labels)
            batch_loss += loss.item()
            total_loss += loss.item()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters() , 1.0)

            optimizer.step()
            scheduler.step()
            
            if(step % 20 == 0 and step != 0) or (step == len(train_dataloader) - 1):

                print(f"{epoch + 1:^7} | {step:^7} | {batch_loss / batch_count:^12.6f} | {'-':^10} | {'-':^9}")

                batch_loss , batch_count = 0 , 0

        avg_train_loss = total_loss / len(train_dataloader)

        train_loss_epochs.append(avg_train_loss)

        print("-"*70)

        if evaluation == True:

            val_loss , val_accuracy = evaluate(model , device , loss_fn , val_dataloader)
            
            val_loss_epochs.append(val_loss)
            val_accuracy_epochs.append(val_accuracy)

            if val_loss < best_val_loss:
                print("Saving the model")
                print(f"Validation loss decreased: ( {best_val_loss} -> {val_loss} )")
                best_val_loss = val_loss
                print(f"Validation Accuracy : {val_accuracy}")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'train_loss': avg_train_loss,
                    }, f"./models/{file_name}_{model_name}_{args.summary_type}_{args.column}_{args.tkl}_{args.batch_size}_{args.lr}.pt")
                
            print(f"{epoch + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss :^10.6f} | {val_accuracy:^9.2f}")
            print("-" * 70)
        
        
        if early_stopping.early_stop_check(val_loss):
            print("Early stopping")
            break

        lr_scheduler.step(val_loss)

        print("\n")
    
    torch.save({'val_accuracy': val_accuracy_epochs , 'val_loss' : val_loss_epochs , 'train_loss' : train_loss_epochs} , f"./metrics/{file_name}_{model_name}_{args.summary_type}_{args.column}_{args.tkl}_{args.batch_size}_{args.lr}.pt")
    print("Training complete")

def evaluate(model , device , loss_fn , val_dataloader):
    """
    Evaluate performance on the validation set
    """
    model.eval()

    val_accuracy = []
    val_loss = []

    for batch in val_dataloader:

        val_input_ids , val_attn_mask , val_labels = tuple( x.to(device) for x in batch)
        val_labels = val_labels.type(torch.LongTensor).to(device)
        
        with torch.no_grad():
            val_logits = model(val_input_ids , val_attn_mask)


        validation_loss = loss_fn(val_logits, val_labels)
        val_loss.append(validation_loss.item())

        preds = torch.argmax(val_logits , dim = 1).flatten()
   
        validation_accuracy = (preds == val_labels).cpu().numpy().mean() * 100
        val_accuracy.append(validation_accuracy)

    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)

    return val_loss , val_accuracy








