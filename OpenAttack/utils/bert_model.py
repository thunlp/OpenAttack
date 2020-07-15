import numpy as np 
from ..classifier import Classifier

class BertModel():
    def __init__(self, model_path, num_labels, max_len = 100, device="cpu"):
        import transformers
        self.tokenizer = transformers.BertTokenizer.from_pretrained(model_path)
        self.device = device
        self.model = transformers.BertForSequenceClassification.from_pretrained(model_path, num_labels=num_labels,output_hidden_states=False)
        self.model.eval()
        self.model.to(self.device)
        self.hook = self.model.bert.embeddings.word_embeddings.register_forward_hook(self.__hook_fn)
        self.max_len = max_len
    
    def to(self, device):
        self.device = device
        self.model.to(self.device)
        return self

    def __hook_fn(self, module, input_, output_):
        self.curr_embedding = output_
        output_.retain_grad()

    def tokenize_corpus(self,corpus):
        tokenized_list = []
        attention_masks = []
        sent_lens = []
        for i in range(len(corpus)):
            sentence = corpus[i]
            result = self.tokenizer.encode_plus(sentence,max_length = self.max_len,pad_to_max_length = True,return_attention_mask = True, truncation=True)
            sent_lens.append( sum(result["attention_mask"]) - 2 )
            sentence_ids = result['input_ids']
            mask = result['attention_mask']
            attention_masks.append(mask)
            tokenized_list.append(sentence_ids)
        return np.array(tokenized_list),np.array(attention_masks), sent_lens

    def predict(self,sen_list, labels=None):
        import torch
        tokeinzed_sen, attentions, sent_lens = self.tokenize_corpus(sen_list)
        result = []
        result_grad = []
        
        if labels is None:
            labels = [0] * len(sen_list)
        labels = torch.LongTensor(labels).to(self.device)

        for i in range(len(tokeinzed_sen)):
            curr_sen = tokeinzed_sen[i]
            curr_mask = attentions[i]
            xs = torch.LongTensor([curr_sen]).to(self.device)
            masks = torch.LongTensor([curr_mask]).to(self.device)
            
            loss, logits = self.model(input_ids = xs,attention_mask = masks, labels=labels[i:i+1])
            logits = torch.nn.functional.softmax(logits,dim=-1)
            loss = - loss
            loss.backward()
            result_grad.append(self.curr_embedding.grad[0].clone())
            result.append(logits.cpu().detach().numpy()[0])
            self.curr_embedding.grad.zero_()

        max_len = max(sent_lens)
        result = np.array(result)
        result_grad = torch.stack(result_grad).cpu().numpy()[:, 1:1 + max_len]
        return result, result_grad

class BertClassifier(Classifier):
    def __init__(self, model_path, num_labels, max_len = 100, device="cpu"):
        self.__model = BertModel(model_path, num_labels, max_len, device)
    
    def to(self, device):
        self.__model.to(device)
        return self
    
    def get_grad(self, input_, labels):
        return self.__model.predict(input_, labels)