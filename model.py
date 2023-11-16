import torch.nn as nn
from transformers import BertModel , AutoModelForSequenceClassification 

class BertClassifier(nn.Module):
    """
    Bert Model 
    """
    def __init__(self, model_name , num_classes = 2, freeze_bert = False):
        """
        @param freeze_bert (bool) : Set 'False' to fine tune the BERT model 
        """
        super(BertClassifier , self).__init__()
        
        # Loading the pre-trained BERT
        self.bert = BertModel.from_pretrained(model_name)
        
        # Defining the hidden layers
        self.hidden_layers = self.bert.config.to_dict()['hidden_size']
        
        # Defining the number of classes
        self.num_classes = num_classes

        # Adding the final linear layer 
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_layers, self.num_classes)
            # nn.ReLU(),
            # nn.Dropout(0.2), 
            # nn.Linear(50 , self.num_classes)
            )
          
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
    
    def forward(self , input_ids , attention_mask):
        """
        Feed forward inputs to BERT

        """
        # Output from the previous layers
        output = self.bert(input_ids = input_ids , attention_mask = attention_mask)

        # Getting the last hidden state
        last_hidden_state_cls = output[0][:, 0, :]

        # Passing through the linear layer
        logits = self.classifier(last_hidden_state_cls)
       
        return logits
    
class finBertClassifier(nn.Module):
    """
    Bert Model 
    """

    def __init__(self, model_name , num_classes = 2, freeze_bert = False):
        """
        @param freeze_bert (bool) : Set 'False' to fine tune the BERT model 
        """
        super(finBertClassifier , self).__init__()
        
        # Loading the pre-trained BERT
        self.bert = AutoModelForSequenceClassification.from_pretrained(model_name , num_labels = 3)

        # Defining the hidden layers
        self.hidden_layers = self.bert.config.to_dict()['hidden_size']
        
        # Defining the number of classes
        self.num_classes = num_classes

        # Adding the final linear layer 
        self.bert.classifier = nn.Linear(self.hidden_layers, self.num_classes)
           
       
    def forward(self , input_ids , attention_mask):
        """
        Feed forward inputs to BERT

        """
        # Output from the previous layers
        output = self.bert(input_ids = input_ids , attention_mask = attention_mask)

        # Getting the last output layer
        output = output[0]
        
        return output

