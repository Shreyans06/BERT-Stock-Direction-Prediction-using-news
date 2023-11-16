import re
import torch
from summary import summarize

def text_preprocessing(text , summary_type):
    """
    - Remove entity mentions (eg. '@united')
    - Correct errors (eg. '&amp;' to '&')
    @param    text (str): a string to be processed.
    @return   text (str): the processed string.
    """
    # text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('\n', '', text)
    
    # Remove '@name'
    text = re.sub(r'(@.*?)[\s]', ' ', text)

    # Replace '&amp;' with '&'
    text = re.sub(r'&amp;', '&', text)

    # Remove trailing whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    if summary_type == "spacy":
        words = text.split(" ")
        per = 200 / len(words) 
        text = summarize(text , per)
        
    return text

def bert_preprocessing(tokenizer , text , MAX_LEN = 512):
    """
    Perform pre-processing for pre-Trained BERT
    @param    tokenizer(object) : Type of tokenizer to be used
    @param    text(list) : Array of texts to be pre - processed
    @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
    @return   attention_masks (torch.Tensor): Tensor of indices specifying which
                  tokens should be attended to by the model.
    """

    input_ids = []
    attention_masks = []
    for index , row in enumerate(text):
        
        encoded = tokenizer.__call__(text = row , add_special_tokens = True , padding = 'max_length' , truncation = True , max_length = MAX_LEN , return_attention_mask = True)
        
        input_ids.append(encoded.get('input_ids'))
        attention_masks.append(encoded.get('attention_mask'))
        
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids , attention_masks
