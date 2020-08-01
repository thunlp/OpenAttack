import numpy as np
from copy import deepcopy
from ..attacker import Attacker
from ..data_manager import DataManager
from ..text_processors import DefaultTextProcessor


def get_min(indices_adv1, d):
    d1 = deepcopy(d)
    idx_adv1 = indices_adv1[np.argmin(d1[indices_adv1])]
    orig_idx_adv1 = idx_adv1
    return orig_idx_adv1


DEFAULT_CONFIG = {
    "sst": False,
    "processor": DefaultTextProcessor(),
}


class GANAttacker(Attacker):
    def __init__(self, **kwargs):
        """
        :param bool sst: Use model trained on sst-2 dataset or snli. True: use sst-2 **Default:** False
        :param processor: text processor used in this attacker. **Default:** :any:`text_processors.DefaultTextProcessor`

        :Package Requirements:
            * torch
        :Data Requirements: :py:data:`.AttackAssist.GAN` :py:data:`.AttackAssist.SGAN`
        :Classifier Capacity: Probability

        Generating Natural Adversarial Examples. Zhengli Zhao, Dheeru Dua, Sameer Singh. ICLR 2018.
        `[pdf] <https://arxiv.org/pdf/1710.11342.pdf>`__
        `[code] <https://github.com/zhengliz/natural-adversary>`__
        """
        self.config = DEFAULT_CONFIG.copy()
        self.config.update(kwargs)
        if self.config['sst'] is False:  # snli
            self.word2idx, self.autoencoder, self.inverter, self.gan_gen, self.gan_disc = DataManager.load("AttackAssist.GAN")
            self.maxlen = 10
        else:
            self.word2idx, self.autoencoder, self.inverter, self.gan_gen, self.gan_disc = DataManager.load("AttackAssist.SGAN")
            self.maxlen = 100
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.gan_gen = self.gan_gen.cpu()
        self.inverter = self.inverter.cpu()
        self.autoencoder.eval()
        self.autoencoder = self.autoencoder.cpu()
        self.right = 0.05  # ####
        self.nsamples = 10
        self.autoencoder.gpu = False
        self.lowercase = True


    def __call__(self, clsf, hypothesis_orig, target=None):
        if self.config['sst'] is False:
            return self.snli_call(clsf, hypothesis_orig, target=target)
        else:
            return self.sst_call(clsf, hypothesis_orig, tt=target)


    def snli_call(self, clsf, hypothesis_orig, target=None):
        import torch
        from torch.autograd import Variable

        # * **clsf** : **Classifier** .
        # * **x_orig** : Input sentence.
        # 'entailment': 0, 'neutral': 1, 'contradiction': 2

        y_orig = clsf.get_pred([hypothesis_orig])[0]
        # tokenization
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
        c = self.autoencoder.encode(torch.tensor([hypothesis, hypothesis], dtype=torch.long),
                                    torch.tensor([length, length], dtype=torch.long), noise=False)
        z = self.inverter(c).data.cpu()

        hypothesis = torch.tensor(hypothesis, dtype=torch.long)
        hypothesis = hypothesis.unsqueeze(0)
        right_curr = self.right
        counter = 0

        while counter <= 5:
            mus = z.repeat(self.nsamples, 1)
            delta = torch.FloatTensor(mus.size()).uniform_(-1 * right_curr, right_curr)
            dist = np.array([np.sqrt(np.sum(x ** 2)) for x in delta.cpu().numpy()])
            perturb_z = Variable(mus + delta)  # ####  volatile=True

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
                adv_prob.append(clsf.get_pred([self.config["processor"].detokenizer(words)])[0])
                sentences.append(self.config["processor"].detokenizer(words))
            for i in range(self.nsamples):
                if target is None:
                    if adv_prob[i] != y_orig:
                        index_adv.append(i)
                else:
                    if int(adv_prob[i]) is int(target):
                        index_adv.append(i)

            if len(index_adv) == 0:
                counter += 1
                right_curr *= 2
            else:
                idx_adv = get_min(index_adv, dist)
                return sentences[idx_adv], clsf.get_pred([sentences[idx_adv]])[0]
        return None

    def sst_call(self, clsf, hypothesis_orig, tt=None):
        import torch
        from torch.autograd import Variable

        y_orig = clsf.get_pred([hypothesis_orig])[0]
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
        target_orig = hypothesis[1:]
        if len(source_orig) > self.maxlen:
            source_orig = source_orig[:self.maxlen]
        if len(target_orig) > self.maxlen:
            target_orig = target_orig[:self.maxlen]
        zeros = (self.maxlen - len(target_orig)) * [0]
        source_orig += zeros
        target_orig += zeros

        source = torch.tensor(np.array(source_orig), dtype=torch.long)
        target = torch.tensor(np.array(target_orig), dtype=torch.long).view(-1)

        source = Variable(source)
        target = Variable(target)
        mask = target.gt(0)
        masked_target = target.masked_select(mask)
        ntokens = len(self.word2idx)
        output_mask = mask.unsqueeze(1).expand(mask.size(0), ntokens)
        output = self.autoencoder(torch.tensor([source_orig], dtype=torch.long),
                                  torch.tensor([length], dtype=torch.long),
                                  noise=True)
        flattened_output = output.view(-1, ntokens)  # ####

        masked_output = flattened_output.masked_select(output_mask).view(-1, ntokens)

        max_values, max_indices = torch.max(output, 2)
        max_indices = max_indices.view(output.size(0), -1).data.cpu().numpy()
        target = target.view(output.size(0), -1).data.cpu().numpy()
        for t, idx in zip(target, max_indices):
            words = [self.idx2word[x] for x in idx]
            if "<eos>" in words:
                words = words[:words.index("<eos>")]
            if "." in words:
                words = words[:words.index(".")]
            for i in range(len(words)):
                if words[i] is "<oov>":
                    words[i] = ""
            sent = self.config["processor"].detokenizer(words)
            pred = clsf.get_pred([sent])[0]
            if tt is None:
                if pred != y_orig:
                    return sent, pred
            else:
                if int(pred) == int(tt):
                    return sent, pred
        return None
