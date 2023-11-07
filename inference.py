from numpy import argmax
import torch 
from model import BertClassifier , DistillBERTClass , finBertClassifier
from transformers import AutoTokenizer , AutoModelForSequenceClassification 
import torch.nn.functional as F
import numpy as np
from transformers import pipeline

MAX_LEN = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def bert_predict(model, test_text):
    """Perform a forward pass on the trained BERT model to predict probabilities
    on the test set.
    """
    # Put the model into the evaluation mode.
    
    model.eval()

    all_logits = []

    # Load batch to GPU
    b_input_ids, b_attn_mask = test_text

    b_input_ids = torch.tensor([b_input_ids])
    b_attn_mask = torch.tensor([b_attn_mask])
    
    # Compute logits
    with torch.no_grad():
        logits = model(b_input_ids, b_attn_mask)

    all_logits.append(logits)
    
    # Concatenate logits from each batch
    all_logits = torch.cat(all_logits, dim=0)

    # Apply softmax to calculate probabilities
    probs = F.softmax(all_logits, dim=1).cpu().numpy()

    return probs

def infer(text):
    # bert_classifier = BertClassifier("bert-base-uncased" , num_classes= 2 , freeze_bert = False)
    bert_classifier = finBertClassifier("ProsusAI/finbert")

    checkpoint = torch.load("/home/snola136/SWM/models/amazon_news_hourly_summarized_spacy_ProsusAI_finbert_spacy_text_512_32_5e-05.pt")

    bert_classifier.load_state_dict(checkpoint['model_state_dict'])

    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")

    encoded = tokenizer.__call__(text , add_special_tokens = True , padding = 'max_length' , truncation = True , max_length = MAX_LEN , return_attention_mask = True)
    input_ids = encoded.get('input_ids')
    attn_mask = encoded.get('attention_mask')

    probs = bert_predict(bert_classifier, (input_ids , attn_mask))

    threshold = 0.5
    class_output =  np.where(probs[:, 1] > threshold, 1, 0)
    return class_output[0]


def test(text):
    pipe = pipeline("text-classification", model="ProsusAI/finbert")
    result = pipe(text)[0]['label']
    score = pipe(text)[0]['score'] 

    if result == "neutral" and score > 0.95:
        return 1

    elif result == "positive":
        return 1
    
    return 0

print(test(text))