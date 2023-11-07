import torch.nn as nn
from transformers import BertModel , AutoModelForSequenceClassification , DistilBertModel

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
            nn.Linear(self.hidden_layers, 50),
            nn.ReLU(),
            # nn.Dropout(0.2), 
            nn.Linear(50 , self.num_classes)
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

class DistillBERTClass(nn.Module):
    def __init__(self , num_classes = 2):
        super(DistillBERTClass, self).__init__()
        self.l1 = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.pre_classifier = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output