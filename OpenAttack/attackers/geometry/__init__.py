import numpy as np
import torch
import torch.nn as nn
import copy
from typing import List, Optional
from ...text_process.tokenizer import Tokenizer, get_default_tokenizer
from ...attack_assist.substitute.word import WordSubstitute, get_default_substitute
from ...utils import get_language, check_language, language_by_name
from ..classification import ClassificationAttacker, Classifier
from ...attack_assist.goal import ClassifierGoal
from ...tags import TAG_English, Tag
from ...exceptions import WordNotInDictionaryException
from transformers import BertConfig, BertTokenizer
from transformers import BertForSequenceClassification, BertForMaskedLM
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
import time
# specific to geometry attack
from nltk.corpus import stopwords
import string
from collections import Counter
from copy import deepcopy
from torch.autograd.gradcheck import zero_gradients
from torch.nn import CosineSimilarity
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from tqdm import tqdm

DEFAULT_CONFIG = {
    "threshold": 0.5,
    "substitute": None,
    "token_unk": "<UNK>",
    "token_pad": "<PAD>",
    "mlm_path": 'bert-base-uncased',
    "num_label": 2,
    "k": 5,
    "use_bpe": 0,
    "threshold_pred_score": 0,
    "use_sim_mat": 0,
    "max_length": 50,
    "max_steps": 50,
    "model": 'lstm',
    "embedding": 'random',
    "hidden_size": 128,
    "bidirectional": False,
    "dataset": 'imdb',
    "vocab_size": 60000,
    "attack": 'deepfool',
    "splits": 1500,
    "max_loops": 5,
    "abandon_stopwords": True,
    "metric": 'projection',
    "model_path": 'models/9.pth',
    "embedding_file": 'glove.6B.300d.txt',
    "embedding_size": 100
}

filter_words = ['a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'ain', 'all', 'almost',
                'alone', 'along', 'already', 'also', 'although', 'am', 'among', 'amongst', 'an', 'and', 'another',
                'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are', 'aren', "aren't", 'around', 'as',
                'at', 'back', 'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides',
                'between', 'beyond', 'both', 'but', 'by', 'can', 'cannot', 'could', 'couldn', "couldn't", 'd', 'didn',
                "didn't", 'doesn', "doesn't", 'don', "don't", 'down', 'due', 'during', 'either', 'else', 'elsewhere',
                'empty', 'enough', 'even', 'ever', 'everyone', 'everything', 'everywhere', 'except', 'first', 'for',
                'former', 'formerly', 'from', 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'he', 'hence',
                'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 'himself', 'his',
                'how', 'however', 'hundred', 'i', 'if', 'in', 'indeed', 'into', 'is', 'isn', "isn't", 'it', "it's",
                'its', 'itself', 'just', 'latter', 'latterly', 'least', 'll', 'may', 'me', 'meanwhile', 'mightn',
                "mightn't", 'mine', 'more', 'moreover', 'most', 'mostly', 'must', 'mustn', "mustn't", 'my', 'myself',
                'namely', 'needn', "needn't", 'neither', 'never', 'nevertheless', 'next', 'no', 'nobody', 'none',
                'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'o', 'of', 'off', 'on', 'once', 'one', 'only',
                'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'per',
                'please', 's', 'same', 'shan', "shan't", 'she', "she's", "should've", 'shouldn', "shouldn't", 'somehow',
                'something', 'sometime', 'somewhere', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs',
                'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein',
                'thereupon', 'these', 'they', 'this', 'those', 'through', 'throughout', 'thru', 'thus', 'to', 'too',
                'toward', 'towards', 'under', 'unless', 'until', 'up', 'upon', 'used', 've', 'was', 'wasn', "wasn't",
                'we', 'were', 'weren', "weren't", 'what', 'whatever', 'when', 'whence', 'whenever', 'where',
                'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while',
                'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'with', 'within', 'without', 'won',
                "won't", 'would', 'wouldn', "wouldn't", 'y', 'yet', 'you', "you'd", "you'll", "you're", "you've",
                'your', 'yours', 'yourself', 'yourselves']
filter_words = set(filter_words)


class Sample:
    def __init__(self, data, words, steps, label, length, id):
        self.word_ids = data[0:steps]
        if words is not None:
            self.sentence = words[0:steps]
        self.length = length
        self.label = label
        self.id = id
        self.history = []
        self.new_info = None

        self.mask = None
        self.stopwords_mask = None

        self.stopwords = set(stopwords.words('english'))
        self.punctuations = string.punctuation

    def set_mask(self, mask, stopwords_mask):
        self.mask = mask
        self.stopwords_mask = stopwords_mask

    def set_new_info(self, new_info):
        # [new_id, new_word, old_id, old_word, idx]
        self.new_info = new_info

class DeepFool(nn.Module):
    def __init__(self, config, num_classes, max_iters, overshoot=0.02):
        super(DeepFool, self).__init__()
        self.config = config
        self.num_classes = num_classes
        self.loops_needed = None

        self.max_iters = max_iters
        self.overshoot = overshoot
        self.loops = 0

    def forward(self, vecs, net_, target=None):
        """

        :param vecs: [batch_size, vec_size]
        :param net_: FFNN in our case
        :param target:
        :return:
        """

        net = deepcopy(net_.classifier)
        sent_vecs = deepcopy(vecs.data)
        input_shape = sent_vecs.size()
        
        f_vecs = net.forward(sent_vecs).data
        # print("input and output:", sent_vecs.shape, f_vecs.shape)
        I = torch.argsort(f_vecs, dim=1, descending=True)
        # I = torch.argsort(f_vecs, dim=-1, descending=True)
        I = I[:, 0:self.num_classes]
        # print("I shape:", I.shape)
        # this is actually the predicted label
        label = I[:, 0]
        # print("label:", label)

        if target is not None:
            I = target.unsqueeze(1)
            if self.config['dataset'] == 'imdb':
                num_classes = 2
            elif self.config['dataset'] == 'agnews':
                num_classes = 4
            else:
                print('Unrecognized dataset {}'.format(self.config['dataset']))
        else:
            num_classes = I.size(1)

        pert_vecs = deepcopy(sent_vecs)
        r_tot = torch.zeros(input_shape)
        check_fool = deepcopy(sent_vecs)

        k_i = label
        loop_i = 0

        # pre-define an finish_mask, [batch_size], all samples are not finished at first
        finish_mask = torch.zeros((input_shape[0], 1), dtype=torch.float)
        finished = torch.ones_like(finish_mask)
        self.loops_needed = torch.zeros((input_shape[0],))

        if torch.cuda.is_available():
            r_tot = r_tot.cuda()
            finish_mask = finish_mask.cuda()
            finished = finished.cuda()
            self.loops_needed = self.loops_needed.cuda()

        # every sample needs to be finished, and total loops should be smaller than max_iters
        while torch.sum(finish_mask >= finished) != input_shape[0] and loop_i < self.max_iters:
            x = pert_vecs.requires_grad_(True)
            fs = net.forward(x)

            pert = torch.ones(input_shape[0])*np.inf
            w = torch.zeros(input_shape)

            if torch.cuda.is_available():
                pert = pert.cuda()
                w = w.cuda()

            # fs[sample_index, I[sample_index, sample_label]]
            logits_label_sum = torch.gather(
                fs, dim=1, index=label.unsqueeze(1)).sum()
            logits_label_sum.backward(retain_graph=True)
            grad_orig = deepcopy(x.grad.data)

            for k in range(1, num_classes):
                if target is not None:
                    k = k - 1
                    if k > 0:
                        break

                zero_gradients(x)
                # fs[sample_index, I[sample_index, sample_class]]
                logits_class_sum = torch.gather(
                    fs, dim=1, index=I[:, k].unsqueeze(1)).sum()
                logits_class_sum.backward(retain_graph=True)

                # [batch_size, n_channels, height, width]
                cur_grad = deepcopy(x.grad.data)
                w_k = cur_grad - grad_orig

                # fs[sample_index, I[sample_index, sample_class]] - fs[sample_index, I[sample_index, sample_label]]
                f_k = torch.gather(fs, dim=1, index=I[:, k].unsqueeze(
                    1)) - torch.gather(fs, dim=1, index=label.unsqueeze(1))
                f_k = f_k.squeeze(-1)

                # element-wise division
                pert_k = torch.div(torch.abs(f_k), self.norm_dim(w_k))

                valid_pert_mask = pert_k < pert

                new_pert = pert_k + 0.
                new_w = w_k + 0.

                valid_pert_mask = valid_pert_mask.bool()
                pert = torch.where(valid_pert_mask, new_pert, pert)
                # index by valid_pert_mask
                valid_w_mask = torch.reshape(
                    valid_pert_mask, shape=(input_shape[0], 1)).float()
                valid_w_mask = valid_w_mask.bool()

                w = torch.where(valid_w_mask, new_w, w)

            r_i = torch.mul(torch.clamp(pert, min=1e-4).reshape(-1, 1), w)
            r_i = torch.div(r_i, self.norm_dim(w).reshape((-1, 1)))

            r_tot_new = r_tot + r_i

            # if get 1 for cur_update_mask, then the sample has never changed its label, we need to update it
            cur_update_mask = (finish_mask < 1.0).byte()
            if torch.cuda.is_available():
                cur_update_mask = cur_update_mask.cuda()

            cur_update_mask = cur_update_mask.bool()

            r_tot = torch.where(cur_update_mask, r_tot_new, r_tot)

            # r_tot already filtered with cur_update_mask, no need to do again
            pert_vecs = sent_vecs + r_tot
            check_fool = sent_vecs + (1.0 + self.overshoot) * r_tot

            k_i = torch.argmax(net.forward(
                check_fool.requires_grad_(True)), dim=-1).data

            if target is None:
                # in untargeted version, we finish perturbing when the network changes its predictions to the advs
                finish_mask += ((k_i != label)*1.0).reshape((-1, 1)).float()
                # print(torch.sum(finish_mask >= finished))
            else:
                # in targeted version, we finish perturbing when the network classifies the advs as the target class
                finish_mask += ((k_i == target)*1.0).reshape((-1, 1)).float()

            loop_i += 1
            self.loops += 1
            self.loops_needed[cur_update_mask.squeeze()] = loop_i

            r_tot.detach_()
            check_fool.detach_()
            r_i.detach_()
            pert_vecs.detach_()

        # grad is not really need for deepfool, used here as an additional check
        x = pert_vecs.requires_grad_(True)
        fs = net.forward(x)

        torch.sum(torch.gather(fs, dim=1, index=k_i.unsqueeze(
            1)) - torch.gather(fs, dim=1, index=label.unsqueeze(1))).backward(retain_graph=True)

        grad = deepcopy(x.grad.data)
        grad = torch.div(grad, self.norm_dim(grad).unsqueeze(1))

        label = deepcopy(label.data)

        if target is not None:
            # in targeted version, we move an adv towards the true class, but we do not want to cross the boundary
            pert_vecs = deepcopy(pert_vecs.data)
            return grad, pert_vecs, label
        else:
            # check_fool should be on the other side of the decision boundary
            check_fool_vecs = deepcopy(check_fool.data)
            return grad, check_fool_vecs, label

    @staticmethod
    def norm_dim(w):
        norms = []
        for idx in range(w.size(0)):
            norms.append(w[idx].norm())
        norms = torch.stack(tuple(norms), dim=0)

        return norms


class WordSaliencyBatch:
    def __init__(self, config, word2id):
        self.config = config
        self.word2id = word2id
        self.UNK_WORD = self.config['token_unk']

        self.model = None

    def split_forward(self, new_word_ids, new_lengths):

        # split new_word_ids and new_lengths
        new_word_ids_splits = new_word_ids.split(1, dim=0)
        new_lengths_splits = new_lengths.split(1, dim=0)
        new_logits = []

        for idx in range(len(new_lengths_splits)):
            outputs = self.model(new_word_ids_splits[idx])
            new_logits_split = outputs.logits
            new_logits.append(new_logits_split)

        new_logits = torch.cat(new_logits, dim=0)
        return new_logits

    def compute_saliency(self, model_, word_ids, labels, lengths, mask, order=False):
        """
        compute saliency for a batch of examples
        # TODO: implement batch to more than one examples
        :param model_:
        :param word_ids: [batch_size, max_steps]
        :param labels: [batch_size]
        :param lengths: [batch_size]
        :param mask: [batch_size, max_steps]
        :param order:
        :return:
        """
        # with torch.no_grad():
        print('start')
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_max_memory_cached()
        print(torch.cuda.max_memory_allocated()/1024/1024)
        print(torch.cuda.memory_allocated()/1024/1024)
        self.model = deepcopy(model_)
        

        # self.model.eval()

        # self.model = model_
        # cur_batch_size = word_ids.size(0)
        cur_batch_size = 1

        unk_id = self.word2id[self.UNK_WORD]
        unk_id = torch.tensor(unk_id)
        if torch.cuda.is_available():
            unk_id = unk_id.cuda()

        # compute the original probs for true class
        # logits: [batch_size, num_classes]
        # predictions: [batch_size]
        # probs: [batch_size, num_classes]
        # logits, _ = self.model(word_ids, lengths)
        logits = self.model(word_ids).logits
        predictions = torch.argmax(logits, dim=-1)
        probs = torch.softmax(logits, dim=-1)

        # [batch_size, num_classes]
        one_hot_mask = torch.arange(logits.size(1)).unsqueeze(
            0).repeat(cur_batch_size, 1)
        if torch.cuda.is_available():
            one_hot_mask = one_hot_mask.cuda()

        one_hot_mask = one_hot_mask == predictions.unsqueeze(1)
        # [batch_size, 1]
        true_probs = torch.masked_select(probs, one_hot_mask)
        # unsqueeze word_ids
        # [batch_size, 1, max_steps]
        new_word_ids = word_ids.unsqueeze(0)
        # new_word_ids = new_word_ids.unsqueeze(1)

        # print("before word id shape:", new_word_ids.shape)
        # [batch_size, max_steps, max_steps]
        # dim 1 used to indicate which word is replaced by unk
        new_word_ids = new_word_ids.repeat(1, self.config['max_steps'], 1)

        # then replace word by unk
        # [max_steps, max_steps]
        # diagonal elements = 1
        diag_mask = torch.diag(torch.ones(self.config['max_steps']))

        # [1, max_steps, max_steps]
        diag_mask = diag_mask.unsqueeze(0)

        # [batch_size, max_steps, max_steps]
        # for elements with a mask of 1, replace with unk_id
        diag_mask = diag_mask.repeat(cur_batch_size, 1, 1).bool()
        if torch.cuda.is_available():
            diag_mask = diag_mask.cuda()
        # [batch_size, max_steps, max_steps]
        # replace with unk_id
        # print(diag_mask.shape, unk_id.shape,new_word_ids.shape)
        new_word_ids = diag_mask * unk_id + (~diag_mask) * new_word_ids

        # compute probs for new_word_ids
        # [batch_size*max_steps, max_steps]
        new_word_ids = new_word_ids.view(
            cur_batch_size*self.config['max_steps'], -1)

        # construct new_lengths
        # [batch_size, 1]
        new_lengths = torch.ones([cur_batch_size, 1]) * lengths
        # print("size 1:", new_lengths.size())
        # new_lengths = lengths.view(cur_batch_size, 1)

        # new_lengths = torch.ones((cur_batch_size, 1)) * lengths
        # repeat
        # [batch_size*max_steps]
        new_lengths = new_lengths.repeat(
            1, self.config['max_steps']).view(-1)
        # print("size 2:", new_lengths.size())

        # the same applies to new_predictions
        # [batch_size, 1]
        new_predictions = predictions.view(cur_batch_size, 1)
        # repeat
        # [batch_size*max_steps]
        new_predictions = new_predictions.repeat(
            1, self.config['max_steps']).view(-1)

        # [batch_size*max_steps, num_classes]
        one_hot_mask = torch.arange(logits.size(1)).unsqueeze(
            0).repeat(new_predictions.size(0), 1)
        if torch.cuda.is_available():
            one_hot_mask = one_hot_mask.cuda()

        one_hot_mask = one_hot_mask == new_predictions.unsqueeze(1)

        # [batch_size*max_steps, num_classes]
        # new_logits, _ = self.model(new_word_ids, new_lengths)
        # start = timer()
        new_logits = self.split_forward(new_word_ids, new_lengths)
        # end = timer()
        # print('time = {}'.format(end - start))
        # sys.stdout.flush()
        # [batch_size*max_steps, num_classes]
        all_probs = torch.softmax(new_logits, dim=-1)

        #print('end')
        #print(torch.cuda.max_memory_allocated()/1024/1024)
        #print(torch.cuda.memory_allocated()/1024/1024)

        # [batch_size, max_steps]
        all_true_probs = torch.masked_select(
            all_probs, one_hot_mask).view(cur_batch_size, -1)
        # only words with a mask of 1 will be considered
        # setting the prob of unqualified words to a large number
        all_true_probs[~mask] = 100.0

        if torch.cuda.is_available():
            all_true_probs = all_true_probs.cuda()
        # [batch_size, max_steps]
        saliency = true_probs.unsqueeze(1) - all_true_probs
        # select the word with the largest saliency
        # [batch_size]
        best_word_idx = torch.argmax(saliency, dim=1)
        replace_order = torch.argsort(saliency, descending=True)
        # check
        check = (best_word_idx < lengths).sum().data.cpu().numpy()

        # assert check == cur_batch_size

        if order:
            return best_word_idx, replace_order
        else:
            return best_word_idx


class GreedyAttack:
    """
    Select words greedily as an attack
    """

    def __init__(self, config, word2id, id2word, vocab, wordid2synonyms):
        self.config = config
        self.stopwords = set(stopwords.words('english'))
        self.mode = None
        self.samples = None
        self.cosine_similarity = CosineSimilarity(dim=1, eps=1e-6)
        self.global_step = 0
        self.word2id = word2id
        self.id2word = id2word
        self.vocab = vocab
        self.wordid2synonyms = wordid2synonyms
        self.max_loops = self.config['max_loops']

        if self.config['attack'] == 'deepfool':
            # in fact, deepfool will finish far more quicker than this
            self.attack = DeepFool(config, num_classes=2, max_iters=20)
        else:
            print('Attack {} not recognized'.format(self.config['attack']))

        self.model = None
        self.word_saliency = WordSaliencyBatch(config, word2id)

    def select_word_batch(self, all_word_ids, cur_available, labels, lengths, finish_mask, stopwords_mask, mask, previous_replaced_words=None):
        """
        select words in a batch fashion
        :param all_word_ids: [batch_size, max_steps]
        :param cur_available: [batch_size, max_steps]
        :param labels: [batch_size]
        :param lengths: [batch_size]
        :param finish_mask: [batch_size]
        :param stopwords_mask: [batch_size, max_steps]
        :param mask: [batch_size, max_steps]
        :param previous_replaced_words: a list of length batch_size
        :return:
        """

        # currently, batched word_saliency is too mem consuming
        cur_batch_size = 1
        all_replace_orders = []

        # t = self.select_word(word_ids=all_word_ids, cur_available=cur_available,
        # 									 label=labels, length=lengths)
        if self.config['abandon_stopwords']:
            mask = torch.mul(mask, stopwords_mask)

        if torch.cuda.is_available():
            mask = mask.cuda()
            cur_available = cur_available.cuda()
        mask = torch.mul(mask, cur_available)
        mask = mask.bool()
        _, all_replace_orders = self.word_saliency.compute_saliency(model_=self.model, word_ids=all_word_ids,
                                                                    labels=labels, lengths=lengths, mask=mask, order=True)

        if torch.cuda.is_available():
            all_replace_orders = all_replace_orders.cuda()

        # [batch_size, max_steps]
        return all_replace_orders

    def construct_new_sample_batch2(self, word_ids, labels, lengths, word_indices, sample_ids, finish_mask):
        """

        :param word_ids: [batch_size, max_steps]
        :param labels: [batch_size]
        :param lengths: [batch_size]
        :param word_indices: [batch_size], the best word in each example to replace
        :param sample_ids: [batch_size]
        :param finish_mask: [batch_size]
        :return:
        """

        # cur_batch_size = sample_ids.size(0)
        cur_batch_size = 1

        all_new_lengths = []
        all_new_labels = []
        all_new_word_ids = []

        n_new_samples = []

        for idx in range(cur_batch_size):
            new_word_ids, new_lengths, new_labels = self.construct_new_sample2(word_ids=word_ids[idx], label=labels, length=lengths,
                                                                               word_idx=word_indices[idx], sample_id=sample_ids[idx], finish_mask=finish_mask[idx])
            all_new_word_ids.append(new_word_ids)
            all_new_lengths.append(new_lengths)
            all_new_labels.append(new_labels)
            n_new_samples.append(new_labels.size(0))

        all_new_word_ids = torch.cat(all_new_word_ids)
        all_new_lengths = torch.cat(all_new_lengths)
        all_new_labels = torch.cat(all_new_labels)

        return all_new_word_ids, all_new_lengths, all_new_labels, n_new_samples

    def construct_new_sample2(self, word_ids, label, length, word_idx, sample_id, finish_mask):
        """

        :param word_ids:
        :param label:
        :param length:
        :param word_idx:
        :param sample_id:
        :param finish_mask:
        :return: all_new_word_ids, [N, max_steps]
                    all_new_lengths, []
                    all_new_labels
        """
        label = torch.tensor([label])
        length = torch.tensor([length])
        all_new_lengths = []
        all_new_labels = []
        all_new_word_ids = []

        if finish_mask:
            word_ids = word_ids.unsqueeze(0)
            length = length.unsqueeze(0)
            label = label.unsqueeze(0)

            return word_ids, length, label

        old_id = int(word_ids[word_idx].data.cpu().numpy())
        syn_word_ids = self.wordid2synonyms[old_id]
        # if sample_id == 33:
        # 	for i in syn_word_ids:
        # 		w = id2word[i]
        # 		print(w)

        for i in range(len(syn_word_ids)):
            new_id = syn_word_ids[i]
            new_word_ids = deepcopy(word_ids)
            new_word_ids[word_idx] = new_id

            all_new_word_ids.append(new_word_ids)
            all_new_lengths.append(length)
            all_new_labels.append(label)

        all_new_word_ids = torch.stack(all_new_word_ids)
        all_new_lengths = torch.stack(all_new_lengths)
        all_new_labels = torch.stack(all_new_labels)
        # return all_new_word_ids, torch.Tensor([all_new_lengths]), torch.Tensor([all_new_labels])
        return all_new_word_ids, all_new_lengths, all_new_labels

    def adv_attack(self, word_ids, lengths, labels, sample_ids, model, samples, stopwords_mask, mask):
        """
        attack a batch of words
        :param word_ids: [batch_size, max_steps]
        :param lengths: [batch_size]
        :param labels: [batch_size]
        :param sample_ids: [batch_size]
        :param model:
        :param samples:
        :param stopwords_mask: [batch_size, max_steps]
        :param mask: [batch_size, max_steps]
        :return:
        """
        """

        :param word_ids: [batch_size, max_length]
        :param lengths: [batch_size]
        :param labels: [batch_size]
        :param sample_ids: [batch_size]
        :param model: current model, deepcopy before use
        :param mode:
        :return:
        """
        # important, set model to eval mode
        self.model = deepcopy(model)
        # self.model.eval()
        # self.samples = samples

        word_ids = word_ids.unsqueeze(0)
        cur_batch_size = word_ids.size(0)  # 1

        # logits: [batch_size, num_classes]
        # sent_vecs: [batch_size, hidden_size]
        # logits, sent_vecs = model(word_ids, lengths)
        outputs = model(word_ids)
        logits = outputs.logits 
        sent_vecs = torch.mean(outputs.hidden_states[-1], dim=1)

        # preds: [batch_size], original predictions before perturbing
        original_predictions = torch.argmax(logits, dim=-1)
        num_classes = logits.size(1)
        # [batch_size, num_classes]
        # select by original prediction
        one_hot_mask = torch.arange(num_classes).unsqueeze(
            0).repeat(cur_batch_size, 1)
        if torch.cuda.is_available():
            one_hot_mask = one_hot_mask.cuda()
        one_hot_mask = one_hot_mask == original_predictions.unsqueeze(1)

        original_probs = torch.nn.functional.softmax(logits, dim=-1)
        pred_probs = torch.masked_select(original_probs, one_hot_mask)
        intermediate_pred_probs = []
        intermediate_pred_probs.append(pred_probs)

        # # find the boundary point
        # self.model.zero_grad()
        # normals, pert_vecs, all_original_predictions = self.attack(vecs=sent_vecs, net_=model.hidden)
        # # [batch_size, hidden_size]
        # r_tot = pert_vecs - sent_vecs

        cur_available = torch.ones(cur_batch_size, self.config['max_steps'])

        # [batch_size]
        finish_mask = torch.zeros(cur_batch_size).bool()
        cur_projections = torch.zeros(cur_batch_size)
        cur_predictions = deepcopy(original_predictions.data)
        # [batch_size, max_steps]
        cur_word_ids = deepcopy(word_ids)
        # [batch_size, hidden_size]
        cur_sent_vecs = deepcopy(sent_vecs.data)

        if torch.cuda.is_available():
            finish_mask = finish_mask.cuda()
            cur_predictions = cur_predictions.cuda()
            cur_projections = cur_projections.cuda()
            cur_word_ids = cur_word_ids.cuda()
            cur_available = cur_available.cuda()
            cur_sent_vecs = cur_sent_vecs.cuda()

        # print("original:", [self.text_data.id2word[idx.item()] for idx in word_ids[0]])
        intermediate_projections = []
        intermediate_normals = []
        intermediate_cosines = []
        intermediate_distances = []
        # [batch_size, iter_idx]
        intermediate_word_ids = []
        intermediate_update_masks = []

        # all_word_ids, cur_available, labels, lengths, finish_mask
        # [batch_size, max_steps]

        # all_replace_orders = self.select_word_batch(all_word_ids=word_ids, cur_available=cur_available,
        # 											labels=labels, lengths=lengths, finish_mask=finish_mask,
        # 											stopwords_mask=stopwords_mask, mask=mask)
        previous_replaced_words = []
        intermediate_word_ids.append(word_ids)
        
        for iter_idx in range(self.max_loops):
            #print(f'Running the {iter_idx}th loop...')

            if finish_mask.sum() == cur_batch_size:
                break

            self.model.zero_grad()
            # for cur_samples, find boundary point
            # cur_normals: [batch_size, hidden_size]
            # cur_pert_vecs: [batch_size, hidden_size]
            # cur_original_predictions: [batch_size]
            cur_normals, cur_pert_vecs, cur_original_predictions = self.attack(
                vecs=cur_sent_vecs, net_=self.model)
            intermediate_normals.append(cur_normals)

            # [batch_size, hidden_size]
            cur_r_tot = cur_pert_vecs - cur_sent_vecs
            # print("cur r total:", cur_r_tot)
            # [batch_size], distances to decision boundary
            cur_r_tot_distance = self.norm_dim(cur_r_tot)
            intermediate_distances.append(cur_r_tot_distance)

            # words_to_replace: [batch_size]
            # cur_available: [batch_size, max_steps]
            # cur_available is updated in selected_word_batch
            all_replace_orders = self.select_word_batch(all_word_ids=cur_word_ids, cur_available=cur_available,
                                                        labels=labels, lengths=lengths, finish_mask=finish_mask,
                                                        stopwords_mask=stopwords_mask, mask=mask)
            words_to_replace = all_replace_orders[:, 0]
            #print("words:", words_to_replace)
            words_to_replace_one_hot = torch.nn.functional.one_hot(
                words_to_replace, num_classes=word_ids.size(1))
            cur_available = torch.mul(
                cur_available, 1-words_to_replace_one_hot)

            # all_new_samples have N samples inside
            # n_new_samples: [batch_size], number of new samples for each old sample
            # def construct_new_sample_batch(self, word_ids, labels, lengths, word_indices, sample_ids, finish_mask):
            # start = timer()
            all_new_word_ids, all_new_lengths, all_new_labels, n_new_samples = self.construct_new_sample_batch2(word_ids=cur_word_ids,
                                                                                                                labels=labels, lengths=lengths,
                                                                                                                word_indices=words_to_replace,
                                                                                                                sample_ids=sample_ids, finish_mask=finish_mask)
            assert all_new_word_ids.size(0) == all_new_labels.size(0)

            if torch.cuda.is_available():
                # [N, max_steps]
                all_new_word_ids = all_new_word_ids.cuda()
                # [N]
                #all_new_lengths = all_new_lengths.cuda()
                #all_new_labels = all_new_labels.cuda()

            # compute new sent_vecs
            # all_new_logits: [N, num_classes]
            # all_new_sent_vectors: [N, hidden_size]
            # all_new_logits, all_new_sent_vectors = model(all_new_word_ids, all_new_lengths)
            outputs = model(all_new_word_ids)
            all_new_logits = outputs.logits
            all_new_sent_vectors = torch.mean(outputs.hidden_states[-1],dim=1)
            # [N]
            all_new_predictions = torch.argmax(all_new_logits, dim=-1)
            # [N, num_classes]
            all_new_probs = torch.softmax(all_new_logits, dim=-1).data

            # get new r_tot
            # [N, hidden_size]
            repeats = torch.tensor(n_new_samples)
            if torch.cuda.is_available():
                repeats = repeats.cuda()

            all_cur_sent_vecs = torch.repeat_interleave(
                cur_sent_vecs, repeats=repeats, dim=0)
            all_cur_normals = torch.repeat_interleave(
                cur_normals, repeats=repeats, dim=0)
            all_new_r_tot = all_new_sent_vectors - all_cur_sent_vecs

            # [N]
            all_new_r_tot_length = self.norm_dim(all_new_r_tot)
            all_cosines = self.cosine_similarity(
                all_new_r_tot, all_cur_normals)
            all_projections = torch.mul(all_new_r_tot_length, all_cosines)

            # TODO: instead of projections, use nearest point
            if self.config['metric'] != 'projection':
                all_cur_normals, all_cur_pert_vecs, all_cur_original_predictions = self.attack(
                    vecs=all_new_sent_vectors, net_=model)
                # [N, hidden_size]
                all_cur_r_tot = all_cur_pert_vecs - all_cur_sent_vecs

                # [N]
                all_cur_r_tot_distance = self.norm_dim(all_cur_r_tot)
                all_projections = all_cur_r_tot_distance

            # split all_projections to match individual examples
            # list of tensors, list length: [batch_size]
            all_projections_splited = torch.split(
                all_projections, split_size_or_sections=n_new_samples)
            all_new_predictions_splited = torch.split(
                all_new_predictions, split_size_or_sections=n_new_samples)
            all_new_lengths_splited = torch.split(
                all_new_lengths, split_size_or_sections=n_new_samples)
            all_new_labels_splited = torch.split(
                all_new_labels, split_size_or_sections=n_new_samples)
            all_cosines_splited = torch.split(
                all_cosines, split_size_or_sections=n_new_samples)

            # list length: [batch_size]
            # each item in the list is a tensor, which consists of several tensors of length max_steps
            all_new_word_ids_splited = torch.split(
                all_new_word_ids, split_size_or_sections=n_new_samples, dim=0)
            all_new_sent_vectors_splited = torch.split(
                all_new_sent_vectors, split_size_or_sections=n_new_samples, dim=0)
            all_new_probs_splited = torch.split(
                all_new_probs, split_size_or_sections=n_new_samples, dim=0)
            # for each tensor, pick the one with largest projection
            assert len(all_projections_splited) == cur_batch_size
            # [batch_size]
            selected_indices = []
            selected_projections = []
            selected_predictions = []
            selected_cosines = []
            # [batch_size, max_steps]
            selected_word_ids = []

            # [batch_size, hidden_size]
            selected_sent_vecs = []

            selected_new_probs = []

            for i in range(cur_batch_size):
                selected_idx = torch.argmax(all_projections_splited[i])
                selected_projection = torch.max(all_projections_splited[i])
                if self.config['metric'] != 'projection':
                    selected_idx = torch.argmin(all_projections_splited[i])
                    selected_projection = torch.min(all_projections_splited[i])
                selected_prediction = all_new_predictions_splited[i][selected_idx]
                selected_cosine = all_cosines_splited[i][selected_idx]
                selected_word_ids_for_cur_sample = all_new_word_ids_splited[i][selected_idx]
                selected_sent_vec_for_cur_sample = all_new_sent_vectors_splited[i][selected_idx]
                selected_probs_for_cur_sample = all_new_probs_splited[i][selected_idx]

                selected_indices.append(selected_idx)
                selected_projections.append(selected_projection)
                selected_predictions.append(selected_prediction)
                selected_word_ids.append(selected_word_ids_for_cur_sample)
                selected_sent_vecs.append(selected_sent_vec_for_cur_sample)
                selected_cosines.append(selected_cosine)
                selected_new_probs.append(selected_probs_for_cur_sample)

            # [batch_size]
            selected_indices = torch.tensor(selected_indices)
            selected_projections = torch.tensor(selected_projections)
            selected_predictions = torch.tensor(selected_predictions)
            selected_cosines = torch.tensor(selected_cosines)
            # [batch_size, max_steps]
            selected_word_ids = torch.stack(selected_word_ids, 0)
            # [batch_size, hidden_size]
            selected_sent_vecs = torch.stack(selected_sent_vecs, 0)
            # [batch_size, num_classes]
            selected_new_probs = torch.stack(selected_new_probs, 0)

            # [batch_size]
            cur_pred_probs = torch.masked_select(
                selected_new_probs, one_hot_mask)
            intermediate_pred_probs.append(cur_pred_probs)

            if torch.cuda.is_available():
                selected_indices = selected_indices.cuda()
                selected_projections = selected_projections.cuda()
                selected_predictions = selected_predictions.cuda()
                selected_word_ids = selected_word_ids.cuda()
                selected_sent_vecs = selected_sent_vecs.cuda()

            # update cur_projections, cur_predictions, and cur_word_ids by ~finish_mask
            # all unfinished samples need to be updated
            # [batch_size]
            cur_update_mask = ~finish_mask
            cur_update_mask = torch.mul(
                cur_update_mask, selected_projections > 0)

            # torch.where(condition, x, y) → Tensor
            # x if condition else y
            # [batch_size]
            cur_projections = torch.where(
                cur_update_mask, selected_projections, cur_projections)
           
            cur_predictions = torch.where(
                cur_update_mask, selected_predictions, cur_predictions)
            # [batch_size, max_steps]
            cur_word_ids = torch.where(
                cur_update_mask.view(-1, 1), selected_word_ids, cur_word_ids)
            intermediate_word_ids.append(cur_word_ids)
            # [batch_size, hidden_size]
            cur_sent_vecs = torch.where(
                cur_update_mask.view(-1, 1), selected_sent_vecs, cur_sent_vecs)

            cur_sent_vecs.detach_()
            cur_word_ids.detach_()
            cur_projections.detach_()
            cur_predictions.detach_()

            intermediate_projections.append(cur_projections.data)
            intermediate_cosines.append(selected_cosines.data)

            # if torch.cuda.is_available():
            # 	print(torch.cuda.max_memory_allocated())
            # 	print(torch.cuda.memory_allocated())
            # 	sys.stdout.flush()

            # finish if we successfully fool the model
            # [batch_size]
            cur_finish_mask = (selected_predictions != original_predictions)
            intermediate_update_masks.append(cur_finish_mask)
            finish_mask += cur_finish_mask
            finish_mask = finish_mask.bool()

        # for the last sent_vec, calculate its distance to decision boundary
        final_normals, final_pert_vecs, final_original_predictions = self.attack(
            vecs=cur_sent_vecs, net_=model)
        intermediate_normals.append(final_normals)
        # [batch_size, hidden_size]
        final_r_tot = final_pert_vecs - cur_sent_vecs
        # [batch_size], distances to decision boundary
        final_r_tot_distance = self.norm_dim(final_r_tot)
        intermediate_distances.append(final_r_tot_distance)

        # [batch_size, hidden_size]
        final_r_tot = cur_sent_vecs - sent_vecs
        # [batch_size, max_steps]
        final_word_ids = deepcopy(cur_word_ids)

        # [batch_size]
        final_predictions = deepcopy(cur_predictions)
        # [batch_size, n_loops]
        intermediate_cosines = torch.stack(
            intermediate_cosines).transpose(0, 1)
        intermediate_projections = torch.stack(
            intermediate_projections).transpose(0, 1)
        # [batch_size, n_loops+1]
        intermediate_distances = torch.stack(
            intermediate_distances).transpose(0, 1)
        intermediate_pred_probs = torch.stack(
            intermediate_pred_probs).transpose(0, 1)

        # [batch_size, loops+1, max_steps]
        intermediate_word_ids = torch.stack(
            intermediate_word_ids).transpose(0, 1)
        intermediate_normals = torch.stack(
            intermediate_normals).transpose(0, 1)

        if torch.cuda.is_available():
            final_r_tot = final_r_tot.cuda()
            final_word_ids = final_word_ids.cuda()
            final_predictions = final_predictions.cuda()
            intermediate_normals = intermediate_normals.cuda()
            intermediate_cosines = intermediate_cosines.cuda()
            intermediate_projections = intermediate_projections.cuda()
            intermediate_distances = intermediate_distances.cuda()
        return final_r_tot, final_word_ids, final_predictions, intermediate_normals,\
            intermediate_cosines, intermediate_distances, original_predictions, intermediate_word_ids, intermediate_pred_probs

    @staticmethod
    def norm_dim(w):
        norms = []
        for idx in range(w.size(0)):
            norms.append(w[idx].norm())
        norms = torch.stack(tuple(norms), dim=0)

        return norms


class GEOAttacker(ClassificationAttacker):
    @property
    def TAGS(self):
        return { self.__lang_tag,  Tag("get_pred", "victim"), Tag("get_prob", "victim") }

    def __init__(self, 
            tokenizer : Optional[Tokenizer] = None,
            substitute : Optional[WordSubstitute] = None,
            lang = None,
            **kwargs):
        """
        :param float threshold: Threshold used in substitute module. **Default:** 0.5
        :param WordSubstitute substitute: Substitute method used in this attacker.
        :param TextProcessor processor: Text processor used in this attacker.
        :param str token_unk: A token which means "unknown token" in Classifier's vocabulary.

        :Classifier Capacity: Probability

        Generating Natural Language Adversarial Examples through Probability Weighted Word Saliency. Shuhuai Ren, Yihe Deng, Kun He, Wanxiang Che. ACL 2019.
        `[pdf] <https://www.aclweb.org/anthology/P19-1103.pdf>`__
        `[code] <https://github.com/JHL-HUST/PWWS/>`__
        """
        self.config = DEFAULT_CONFIG.copy()
        self.config.update(kwargs)
        '''
        if self.config["substitute"] is None:
            self.config["substitute"] = WordNetSubstitute()
        '''
        #check_parameters(self.config.keys(), DEFAULT_CONFIG)

        #self.processor = self.config["processor"]
        lst = []
        if tokenizer is not None:
            lst.append(tokenizer)
        if substitute is not None:
            lst.append(substitute)
        if len(lst) > 0:
            self.__lang_tag = get_language(lst)
        else:
            self.__lang_tag = language_by_name(lang)
            if self.__lang_tag is None:
                raise ValueError("Unknown language `%s`" % lang)
        
        if substitute is None:
            substitute = get_default_substitute(self.__lang_tag)
        self.substitute = substitute

        if tokenizer is None:
            tokenizer = get_default_tokenizer(self.__lang_tag)
        self.tokenizer = tokenizer

        #self.substitute = self.config["substitute"]
        self.max_length = self.config["max_length"]
        self.max_steps = self.config["max_steps"]
        self.max_loops = self.config["max_loops"]
        self.UNK_WORD = self.config["token_unk"]
        self.PAD_WORD = self.config["token_pad"]

        self.vocab_size = self.config['vocab_size']
        self.word2id, self.id2word = self.build_vocab(self.config['data'])
        self.vocab = self.get_vocab()
        self.word2synonyms, self.wordid2synonyms = self.construct_synonyms()

        self.embedding_size = self.config['embedding_size']
        # [vocab_size, embed_dim]
        # self.pre_trained_embedding = self.create_embedding()
        self.pre_trained_embedding = None

        self.greedy_attack = GreedyAttack(
            self.config, word2id=self.word2id, id2word=self.id2word, wordid2synonyms=self.wordid2synonyms, vocab=self.vocab)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def preprocess(self, x_batch):
        ls_of_words = [list(map(
            lambda x:x[0], self.tokenizer.tokenize(sent))) for sent in x_batch]
        words = ls_of_words[0]
        seq_len = list(map(lambda x: len(x), x_batch))
        max_len = max(seq_len)
        if self.config["max_len"] is not None:
            max_len = min(max_len, self.config["max_len"])

        words = words[:max_len]
        length = len(words)
        word_ids = []
        for word in words:
            if word in self.word2id.keys():
                id_ = self.word2id[word]
            else:
                id_ = self.word2id[self.config['token_unk']]
            word_ids.append(id_)
        while len(word_ids) < max_len:
            word_ids.append(self.word2id[self.config['token_pad']])
        while len(words) < max_len:
            words.append(self.config['token_pad'])
        return torch.tensor(word_ids), max_len

    def attack(self, clsf: Classifier, x_orig : str, goal : ClassifierGoal):
        #torch.cuda.empty_cache()
        x_orig = x_orig.lower()
        if goal.target is None:
            targeted = False
            target = clsf.get_pred([x_orig])[0]  # calc x_orig's prediction
        else:
            targeted = True

        words = list(
            map(lambda x: x[0], self.tokenizer.tokenize(x_orig)))
        # words = self.config["processor"].get_tokens(x_orig)
        words = words[:self.max_length]
        length = len(words)
        word_ids = []
        for word in words:
            if word in self.word2id.keys():
                id_ = self.word2id[word]
            else:
                id_ = self.word2id[self.UNK_WORD]
            word_ids.append(id_)
        while len(word_ids) < self.max_length:
            word_ids.append(self.word2id[self.PAD_WORD])
        while len(words) < self.max_length:
            words.append(self.PAD_WORD)
        word_ids = [self.word2id[word] for word in words]
        # logits, _ = self.model(torch.tensor(word_ids), length)
        # target = torch.argmax(logits, -1)
        # word_ids, length = clsf.preprocess([x_orig])
        # logits, _ = clsf(torch.tensor(word_ids), length)

        stopwords_mask = []
        sample = Sample(data=word_ids, words=words,
                        steps=self.max_length, label=goal.target, length=length, id=-1)
        self.stopwords = set(stopwords.words('english'))
        self.create_mask(sample, self.stopwords)

        final_word_ids, final_pred = self.inner_attack(clsf.model, sample)
        final_words = [self.id2word[word_id.item()]
                       for word_id in final_word_ids[0]]
        final_sent = self.tokenizer.detokenize(final_words[:length])

        final_pred_clsf = clsf.get_pred([final_sent])[0]
        if final_pred_clsf == goal.target:
            #return final_sent, final_pred_clsf
            return None
        else:
            return final_sent

    def create_mask(self, sample, stopwords):
        mask = []
        stopwords_mask = []
        for idx, word in enumerate(sample.sentence):
            if idx >= sample.length:
                mask.append(0)
                stopwords_mask.append(0)
                continue
            # word = word[0]
            if word in string.punctuation or word not in self.word2id.keys() or word == self.PAD_WORD\
                    or word == self.UNK_WORD or word not in self.vocab:
                mask.append(0)

            elif len(self.word2synonyms[word]) <= 1:
                mask.append(0)
            else:
                mask.append(1)

            if word.lower() in stopwords:
                stopwords_mask.append(0)
            else:
                stopwords_mask.append(1)

            sample.set_mask(mask=mask, stopwords_mask=stopwords_mask)

    def get_vocab(self):
        vocab = []
        for word, idx in self.word2id.items():
            word_ = self.id2word[idx]
            if word_ != word:
                print()
            if word == word_ and not(word == self.PAD_WORD or word == self.UNK_WORD):
                vocab.append(word)
            else:
                continue
        return vocab

    def construct_synonyms(self):
        """
        for each word in the vocab, find its synonyms
        build a dictionary, where key is word, value is its synonyms
        :return:
        """
        word2synonyms, wordid2synonyms = {}, {}
        for word_id in range(len(self.id2word)):
            word = self.id2word[word_id]
            # print("inside construct synonyms:", word)

            if word == self.PAD_WORD or word == self.UNK_WORD:
                word2synonyms[word] = [word]
                wordid2synonyms[word_id] = [word_id]
                continue

            synonyms = []
            synonyms_id = []
            for syn in wordnet.synsets(word):
                for l in syn.lemmas():
                    w = l.name()
                    if w not in self.word2id.keys():
                        continue
                    w_id = self.word2id[w]
                    # if synonym is PAD or UNK, continue
                    if w_id == self.word2id[self.PAD_WORD] or w_id == self.word2id[self.UNK_WORD]:
                        continue
                    synonyms.append(w)
                    synonyms_id.append(w_id)
            # put original word in synonyms
            synonyms.append(word)
            synonyms_id.append(word_id)
            synonyms = list(set(synonyms))
            synonyms_id = list(set(synonyms_id))
            word2synonyms[word] = synonyms

            wordid2synonyms[word_id] = synonyms_id
        return word2synonyms, wordid2synonyms

    def build_vocab(self, data):
        all_words = []
        for elem in data:
            #all_words += self.config["processor"].get_tokens(elem['x'])
            all_words += self.tokenizer.tokenize(elem['x'])
        counter = Counter([word[0].lower() for word in all_words])

        count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

        # keep the most frequent vocabSize words, including the special tokens
        # -1 means we have no limits on the number of words
        if self.vocab_size != -1:
            count_pairs = count_pairs[0:self.vocab_size-2]

        count_pairs.append((self.UNK_WORD, 100000))
        count_pairs.append((self.PAD_WORD, 100000))

        self.vocab_size = min(self.vocab_size, len(count_pairs))

        if self.vocab_size != -1:
            assert len(count_pairs) == self.vocab_size

        words, _ = list(zip(*count_pairs))
        word_to_id = dict(zip(words, range(len(words))))

        id_to_word = {v: k for k, v in word_to_id.items()}
        return word_to_id, id_to_word

    # def load_model(self, model_path):
    #     if torch.cuda.is_available():
    #         model_state_dict, optimizer_state_dict, _, _, _ = torch.load(model_path)
    #     else:
    #         model_state_dict, optimizer_state_dict, _, _, _ = torch.load(model_path, map_location='cpu')
    #     self.model.load_state_dict(model_state_dict)

    def inner_attack(self, model, sample):
        # model.eval()

        if torch.cuda.is_available():
            model.cuda()

        # results = {'original_acc': 0.0, 'acc_perturbed': 0.0, 'change_rate': 0.0, 'n_samples': 0,
        # 		   'original_corrects': 0, 'perturbed_corrects:': 0, 'n_changed': 0, 'n_perturbed': 0}
        # all_replace_rate = []
        # all_n_change_words = []

        if sample.length > self.max_steps:
            length = self.max_steps
        else:
            length = sample.length
        word_ids = sample.word_ids[:self.max_steps]
        stopwords_mask = sample.stopwords_mask[:self.max_steps]
        mask = sample.mask[:self.max_steps]

        sample_ids, word_ids, lengths, labels, stopwords_mask, mask = sample.id, torch.tensor(
            word_ids), length, sample.label, torch.tensor(stopwords_mask), torch.tensor(mask)
        # cur_batch_size = lengths.size(0)
        if torch.cuda.is_available():
            word_ids = word_ids.cuda()
            #lengths = lengths.cuda()
            #labels = labels.cuda()
            stopwords_mask = stopwords_mask.cuda()
            #sample_ids = sample_ids.cuda()
            mask = mask.cuda()

        # [batch_size]
        # perturbed_samples: list of samples
        # perturbed_loops: list of ints
        # perturbed_predictions: tensor
        # original_predictions: tensor
        # perturbed_projections: tensor
        sample_ids = torch.Tensor([sample_ids])
        final_r_tot, final_word_ids, perturbed_predictions, intermediate_normals, intermediate_cosines,\
            intermediate_distances, original_predictions, intermediate_word_ids, intermediate_pred_probs = \
            self.greedy_attack.adv_attack(word_ids=word_ids, lengths=lengths, labels=labels,
                                          sample_ids=sample_ids, model=model, samples=sample,
                                          stopwords_mask=stopwords_mask, mask=mask)
        return final_word_ids, perturbed_predictions
