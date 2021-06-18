import copy
from typing import List, Optional, Union
import numpy as np
from transformers import BertConfig, BertTokenizer, BertForMaskedLM
import torch


from ..classification import ClassificationAttacker, Classifier, ClassifierGoal
from ...tags import TAG_English, Tag
from ...exceptions import WordNotInDictionaryException
from ...attack_assist.substitute.word import get_default_substitute, WordSubstitute
from ...attack_assist.filter_words import get_default_filter_words


class Feature(object):
    def __init__(self, seq_a, label):
        self.label = label
        self.seq = seq_a
        self.final_adverse = seq_a
        self.query = 0
        self.change = 0
        self.success = 0
        self.sim = 0.0
        self.changes = []

class BERTAttacker(ClassificationAttacker):
    @property
    def TAGS(self):
        return { self.__lang_tag, Tag("get_pred", "victim"), Tag("get_prob", "victim") }

    def __init__(self, 
            mlm_path : str = 'bert-base-uncased',
            k : int = 36,
            use_bpe : bool = True,
            sim_mat : Union[None, bool, WordSubstitute] = None,
            threshold_pred_score : float = 0.3,
            max_length : int = 512,
            device : Optional[torch.device] = None,
            filter_words : List[str] = None
        ):
        """
        BERT-ATTACK: Adversarial Attack Against BERT Using BERT, Linyang Li, Ruotian Ma, Qipeng Guo, Xiangyang Xue, Xipeng Qiu, EMNLP2020
        `[pdf] <https://arxiv.org/abs/2004.09984>`__
        `[code] <https://github.com/LinyangLee/BERT-Attack>`__

        Args:
            mlm_path: The path to the masked language model. **Default:** 'bert-base-uncased'
            k: The k most important words / sub-words to substitute for. **Default:** 36
            use_bpe: Whether use bpe. **Default:** `True`
            sim_mat: Whether use cosine_similarity to filter out atonyms. Keep `None` for not using a sim_mat.
            threshold_pred_score: Threshold used in substitute module. **Default:** 0.3
            max_length: The maximum length of an input sentence for bert. **Default:** 512
            device: A computing device for bert.
            filter_words: A list of words that will be preserved in the attack procesudre.

        :Classifier Capacity:
            * get_pred
            * get_prob

        
        """


        self.tokenizer_mlm = BertTokenizer.from_pretrained(mlm_path, do_lower_case=True)
        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        config_atk = BertConfig.from_pretrained(mlm_path)
        self.mlm_model = BertForMaskedLM.from_pretrained(mlm_path, config=config_atk).to(self.device)
        self.k = k
        self.use_bpe = use_bpe
        self.threshold_pred_score = threshold_pred_score
        self.max_length = max_length
        

        self.__lang_tag = TAG_English
        if filter_words is None:
            filter_words = get_default_filter_words(self.__lang_tag)
        self.filter_words = set(filter_words)

        if sim_mat is None or sim_mat is False:
            self.use_sim_mat = False
        else:
            self.use_sim_mat = True
            if sim_mat is True:
                self.substitute = get_default_substitute(self.__lang_tag)
            else:
                self.substitute = sim_mat

    def attack(self, victim: Classifier, sentence, goal: ClassifierGoal):
        x_orig = sentence.lower()

        # return None
        tokenizer = self.tokenizer_mlm
        # MLM-process
        feature = Feature(x_orig, goal.target)
        words, sub_words, keys = self._tokenize(feature.seq, tokenizer)
        max_length = self.max_length
        # original label
        inputs = tokenizer.encode_plus(feature.seq, None, add_special_tokens=True, max_length=max_length, truncation=True)
        input_ids, _ = torch.tensor(inputs["input_ids"]), torch.tensor(inputs["token_type_ids"])

        orig_probs = torch.Tensor(victim.get_prob([feature.seq]))
        orig_probs = orig_probs[0].squeeze()
        orig_probs = torch.softmax(orig_probs, -1)
       
        current_prob = orig_probs.max()

        sub_words = ['[CLS]'] + sub_words[:2] + sub_words[2:max_length - 2] + ['[SEP]']
       
        input_ids_ = torch.tensor([tokenizer.convert_tokens_to_ids(sub_words)])
        word_predictions = self.mlm_model(input_ids_.to(self.device))[0].squeeze()  # seq-len(sub) vocab
        word_pred_scores_all, word_predictions = torch.topk(word_predictions, self.k, -1)  # seq-len k

        word_predictions = word_predictions[1:len(sub_words) + 1, :]
        word_pred_scores_all = word_pred_scores_all[1:len(sub_words) + 1, :]

        important_scores = self.get_important_scores(words, victim, current_prob, goal.target, orig_probs)
        feature.query += int(len(words))
        list_of_index = sorted(enumerate(important_scores), key=lambda x: x[1], reverse=True)
        final_words = copy.deepcopy(words)

        for top_index in list_of_index:
            if feature.change > int(0.2 * (len(words))):
                feature.success = 1  # exceed
                return None

            tgt_word = words[top_index[0]]
            if tgt_word in self.filter_words:
                continue
            if keys[top_index[0]][0] > max_length - 2:
                continue

            substitutes = word_predictions[keys[top_index[0]][0]:keys[top_index[0]][1]]  # L, k
            word_pred_scores = word_pred_scores_all[keys[top_index[0]][0]:keys[top_index[0]][1]]

            substitutes = self.get_substitues(substitutes, tokenizer, self.mlm_model, self.use_bpe, word_pred_scores, self.threshold_pred_score)

            if self.use_sim_mat:
                try:
                    cfs_output = self.substitute(tgt_word)
                    cos_sim_subtitutes = [elem[0] for elem in cfs_output]
                    substitutes = list(set(substitutes) & set(cos_sim_subtitutes))
                except WordNotInDictionaryException:
                    pass
                    # print("The target word is not representable by counter fitted vectors. Keeping the substitutes output by the MLM model.")
            most_gap = 0.0
            candidate = None
            
            for substitute in substitutes:               
                if substitute == tgt_word:
                    continue  # filter out original word
                if '##' in substitute:
                    continue  # filter out sub-word

                if substitute in self.filter_words:
                    continue
                # if substitute in self.w2i and tgt_word in self.w2i:
                #     if self.cos_mat[self.w2i[substitute]][self.w2i[tgt_word]] < 0.4:
                #         continue
                
                temp_replace = final_words
                temp_replace[top_index[0]] = substitute
                temp_text = tokenizer.convert_tokens_to_string(temp_replace)
                inputs = tokenizer.encode_plus(temp_text, None, add_special_tokens=True, max_length=max_length, truncation=True)
                input_ids = torch.tensor(inputs["input_ids"]).unsqueeze(0).to(self.device)
                seq_len = input_ids.size(1)
                
                temp_prob = torch.Tensor(victim.get_prob([temp_text]))[0].squeeze()
                feature.query += 1
                temp_prob = torch.softmax(temp_prob, -1)
                temp_label = torch.argmax(temp_prob)

                if goal.check(feature.final_adverse, temp_label):
                    feature.change += 1
                    final_words[top_index[0]] = substitute
                    feature.changes.append([keys[top_index[0]][0], substitute, tgt_word])
                    feature.final_adverse = temp_text
                    feature.success = 4
                    return feature.final_adverse
                else:
                    label_prob = temp_prob[goal.target]
                    gap = current_prob - label_prob
                    if gap > most_gap:
                        most_gap = gap
                        candidate = substitute

            if most_gap > 0:
                feature.change += 1
                feature.changes.append([keys[top_index[0]][0], candidate, tgt_word])
                current_prob = current_prob - most_gap
                final_words[top_index[0]] = candidate

        feature.final_adverse = (tokenizer.convert_tokens_to_string(final_words))
        feature.success = 2
        return None


    def _tokenize(self, seq, tokenizer):
        seq = seq.replace('\n', '').lower()
        words = seq.split(' ')

        sub_words = []
        keys = []
        index = 0
        for word in words:
            sub = tokenizer.tokenize(word)
            sub_words += sub
            keys.append([index, index + len(sub)])
            index += len(sub)

        return words, sub_words, keys

    def _get_masked(self, words):
        len_text = len(words)
        masked_words = []
        for i in range(len_text - 1):
            masked_words.append(words[0:i] + ['[UNK]'] + words[i + 1:])
        # list of words
        return masked_words
    
    def _get_masked_insert(self, words):
        len_text = len(words)
        masked_words = []
        for i in range(len_text - 1):
            masked_words.append(words[0:i + 1] + ['[UNK]'] + words[i + 1:])
        # list of words
        return masked_words
    
    def get_important_scores(self, words, tgt_model, orig_prob, orig_label, orig_probs):
        masked_words = self._get_masked(words)
        texts = [' '.join(words) for words in masked_words]  # list of text of masked words
        leave_1_probs = torch.Tensor(tgt_model.get_prob(texts))
        leave_1_probs = torch.softmax(leave_1_probs, -1)  
        leave_1_probs_argmax = torch.argmax(leave_1_probs, dim=-1)

        import_scores = (orig_prob
                        - leave_1_probs[:, orig_label]
                        +
                        (leave_1_probs_argmax != orig_label).float()
                        * (leave_1_probs.max(dim=-1)[0] - torch.index_select(orig_probs, 0, leave_1_probs_argmax))
                        ).data.cpu().numpy()

        return import_scores

    def get_bpe_substitues(self, substitutes, tokenizer, mlm_model):
        # substitutes L, k

        substitutes = substitutes[0:12, 0:4] # maximum BPE candidates

        # find all possible candidates 
        all_substitutes = []
        for i in range(substitutes.size(0)):
            if len(all_substitutes) == 0:
                lev_i = substitutes[i]
                all_substitutes = [[int(c)] for c in lev_i]
            else:
                lev_i = []
                for all_sub in all_substitutes:
                    for j in substitutes[i]:
                        lev_i.append(all_sub + [int(j)])
                all_substitutes = lev_i

        # all substitutes  list of list of token-id (all candidates)
        c_loss = torch.nn.CrossEntropyLoss(reduction='none')
        word_list = []
        # all_substitutes = all_substitutes[:24]
        all_substitutes = torch.tensor(all_substitutes) # [ N, L ]
        all_substitutes = all_substitutes[:24].to(self.device)
        # print(substitutes.size(), all_substitutes.size())
        N, L = all_substitutes.size()
        word_predictions = mlm_model(all_substitutes)[0] # N L vocab-size

        ppl = c_loss(word_predictions.view(N*L, -1), all_substitutes.view(-1)) # [ N*L ] 
        ppl = torch.exp(torch.mean(ppl.view(N, L), dim=-1)) # N  
        _, word_list = torch.sort(ppl)
        word_list = [all_substitutes[i] for i in word_list]
        final_words = []
        for word in word_list:
            tokens = [tokenizer._convert_id_to_token(int(i)) for i in word]
            text = tokenizer.convert_tokens_to_string(tokens)
            final_words.append(text)
        return final_words

    def get_substitues(self, substitutes, tokenizer, mlm_model, use_bpe, substitutes_score=None, threshold=3.0):
        # substitues L,k
        # from this matrix to recover a word
        words = []
        sub_len, k = substitutes.size()  # sub-len, k

        if sub_len == 0:
            return words
            
        elif sub_len == 1:
            for (i,j) in zip(substitutes[0], substitutes_score[0]):
                if threshold != 0 and j < threshold:
                    break
                words.append(tokenizer._convert_id_to_token(int(i)))
        else:
            if use_bpe == 1:
                words = self.get_bpe_substitues(substitutes, tokenizer, mlm_model)
            else:
                return words
        return words
    
    def get_sim_embed(self, embed_path, sim_path):
        id2word = {}
        word2id = {}

        with open(embed_path, 'r', encoding='utf-8') as ifile:
            for line in ifile:
                word = line.split()[0]
                if word not in id2word:
                    id2word[len(id2word)] = word
                    word2id[word] = len(id2word) - 1

        cos_sim = np.load(sim_path)
        return cos_sim, word2id, id2word

