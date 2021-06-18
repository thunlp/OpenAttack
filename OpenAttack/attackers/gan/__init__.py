import numpy as np
from copy import deepcopy
from ..classification import ClassificationAttacker, Classifier, ClassifierGoal
from ...data_manager import DataManager
from ...tags import TAG_English, Tag
import torch


def get_min(indices_adv1, d):
    d1 = deepcopy(d)
    idx_adv1 = indices_adv1[np.argmin(d1[indices_adv1])]
    orig_idx_adv1 = idx_adv1
    return orig_idx_adv1


DEFAULT_CONFIG = {
    "sst": False,
}


class GANAttacker(ClassificationAttacker):
    @property
    def TAGS(self):
        return { self.__lang_tag,  Tag("get_pred", "victim") }

    def __init__(self, gan_dataset : str = "sst"):
        """
        Generating Natural Adversarial Examples. Zhengli Zhao, Dheeru Dua, Sameer Singh. ICLR 2018.
        `[pdf] <https://arxiv.org/pdf/1710.11342.pdf>`__
        `[code] <https://github.com/zhengliz/natural-adversary>`__

        Args:
            gan_dataset: The name of dataset which GAN model is trained on. Must be one of the following: ``["sst", "snli"]``. **Default:** sst
        
        :Language: english
        :Classifier Capacity:
            * get_pred

        """
        self.__lang_tag = TAG_English
        self.gan_dataset = gan_dataset
        if  gan_dataset == "snli":  # snli
            self.word2idx, self.autoencoder, self.inverter, self.gan_gen, self.gan_disc = DataManager.load("AttackAssist.GAN")
            self.maxlen = 10
        elif gan_dataset == "sst":
            self.word2idx, self.autoencoder, self.inverter, self.gan_gen, self.gan_disc = DataManager.load("AttackAssist.SGAN")
            self.maxlen = 100
        else:
            raise ValueError("Unknown dataset `%s`" % self.gan_dataset)
        
        self.idx2word = {v: k for k, v in self.word2idx.items()}

        ## TODO: support GPU gan

        self.gan_gen = self.gan_gen.cpu()
        self.inverter = self.inverter.cpu()
        self.autoencoder.eval()
        self.autoencoder = self.autoencoder.cpu()
        self.right = 0.05  # ####
        self.nsamples = 10
        self.autoencoder.gpu = False
        self.lowercase = True


    def attack(self, victim: Classifier, sentence, goal: ClassifierGoal):
        if self.gan_dataset == "snli":
            return self.snli_call(victim, sentence, goal)
        elif self.gan_dataset == "sst":
            return self.sst_call(victim, sentence, goal)
        else:
            raise ValueError("Unknown dataset `%s`" % self.gan_dataset)

    def snli_call(self, clsf : Classifier, hypothesis_orig, goal : ClassifierGoal):

        # * **clsf** : **Classifier** .
        # * **x_orig** : Input sentence.
        # 'entailment': 0, 'neutral': 1, 'contradiction': 2

        # tokenization
        if self.lowercase:
            hypothesis_orig = hypothesis_orig.strip().lower()

        hypothesis_words =  hypothesis_orig.strip().split(" ")
        hypothesis_words = ['<sos>'] + hypothesis_words
        hypothesis_words += ['<eos>']

        vocab = self.word2idx
        unk_idx = vocab['<oov>']
        hypothesis_indices = [vocab[w] if w in vocab else unk_idx for w in hypothesis_words]
        hypothesis_words = [w if w in vocab else '<oov>' for w in hypothesis_words]
        length = min(len(hypothesis_words), self.maxlen)

        if len(hypothesis_indices) < self.maxlen:
            hypothesis_indices += [0] * (self.maxlen - len(hypothesis_indices))
            hypothesis_words += ["<pad>"] * (self.maxlen - len(hypothesis_words))

        hypothesis = hypothesis_indices[:self.maxlen]
        hypothesis_words = hypothesis_words[:self.maxlen]
        c = self.autoencoder.encode(torch.LongTensor([hypothesis, hypothesis]),
                                    torch.LongTensor([length, length]), noise=False)
        z = self.inverter(c).data.cpu()

        hypothesis = torch.LongTensor(hypothesis)
        hypothesis = hypothesis.unsqueeze(0)
        right_curr = self.right
        counter = 0

        while counter <= 5:
            mus = z.repeat(self.nsamples, 1)
            delta = torch.FloatTensor(mus.size()).uniform_(-1 * right_curr, right_curr)
            dist = np.array([np.sqrt(np.sum(x ** 2)) for x in delta.cpu().numpy()])
            perturb_z = mus + delta  # ####  volatile=True

            x_tilde = self.gan_gen(perturb_z)  # perturb
            adv_prob = []
            index_adv = []
            sentences = []
            for i in range(self.nsamples):
                x_adv = x_tilde[i]
                sample_idx = self.autoencoder.generate(x_adv, 10, True).data.cpu().numpy()[0]
                words = [self.idx2word[x] for x in sample_idx]
                if "<eos>" in words:
                    words = words[:words.index("<eos>")]
                adv_prob.append(clsf.get_pred([ " ".join(words) ])[0])
                sentences.append(" ".join(words))
            for i in range(self.nsamples):
                if goal.check(sentences[i], int(adv_prob[i])):
                    index_adv.append(i)

            if len(index_adv) == 0:
                counter += 1
                right_curr *= 2
            else:
                idx_adv = get_min(index_adv, dist)
                return sentences[idx_adv], clsf.get_pred([sentences[idx_adv]])[0]
        return None

    def sst_call(self, clsf : Classifier, hypothesis_orig, target : ClassifierGoal):
        if self.lowercase:
            hypothesis_orig = hypothesis_orig.strip().lower()

        hypothesis_words = hypothesis_orig.strip().split(" ")
        hypothesis_words = ['<sos>'] + hypothesis_words
        hypothesis_words += ['<eos>']

        vocab = self.word2idx
        unk_idx = vocab['<oov>']
        hypothesis_indices = [vocab[w] if w in vocab else unk_idx for w in hypothesis_words]
        hypothesis_words = [w if w in vocab else '<oov>' for w in hypothesis_words]
        length = min(len(hypothesis_words), self.maxlen)

        if len(hypothesis_indices) < self.maxlen:
            hypothesis_indices += [0] * (self.maxlen - len(hypothesis_indices))
            hypothesis_words += ["<pad>"] * (self.maxlen - len(hypothesis_words))

        hypothesis = hypothesis_indices[:self.maxlen]
        hypothesis_words = hypothesis_words[:self.maxlen]
        source_orig = hypothesis[:-1]
        if len(source_orig) > self.maxlen:
            source_orig = source_orig[:self.maxlen]
        zeros = (self.maxlen - len(source_orig)) * [0]
        source_orig += zeros

        ## TODO Something maybe wrong here

        output = self.autoencoder(torch.LongTensor([source_orig]),
                                  torch.LongTensor([length]),
                                  noise=True)

        _, max_indices = torch.max(output, 2)
        max_indices = max_indices.view(output.size(0), -1).data.cpu().numpy()
        for idx in max_indices:
            words = [self.idx2word[x] for x in idx]
            if "<eos>" in words:
                words = words[:words.index("<eos>")]
            if "." in words:
                words = words[:words.index(".")]
            for i in range(len(words)):
                if words[i] == "<oov>":
                    words[i] = ""
            sent = " ".join(words)
            pred = clsf.get_pred([sent])[0]

            if target.check(sent, pred):
                return sent
        return None
