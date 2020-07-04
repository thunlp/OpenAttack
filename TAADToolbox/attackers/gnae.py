from ..attacker import Attacker
from ..data_manager import DataManager
import numpy as np
from copy import deepcopy


DEFAULT_CONFIG = {

}


def get_min(indices_adv1, d):
    d1 = deepcopy(d)
    idx_adv1 = indices_adv1[np.argmin(d1[indices_adv1])]
    orig_idx_adv1 = idx_adv1
    return orig_idx_adv1


class GNAEAttacker(Attacker):
    def __init__(self, **kwargs):
        self.config = DEFAULT_CONFIG.copy()
        self.config.update(kwargs)
        self.idx2word = DataManager.load("Idx2Word")
        self.word2idx = DataManager.load("Word2Idx")
        self.autoencoder = DataManager.load("AutoEncoder")
        self.inverter = DataManager.load("Inverter")
        self.gan_gen = DataManager.load("GanGen")
        self.gan_disc = DataManager.load("GanDisc")
        self.gan_gen = self.gan_gen.cpu()
        self.inverter = self.inverter.cpu()
        self.autoencoder.eval()
        self.autoencoder = self.autoencoder.cpu()
        self.maxlen = 10
        self.right = 0.05
        self.nsamples = 20
        self.autoencoder.gpu = False
        self.lowercase = True
        self.hybrid = False

    def __call__(self, clsf, premise_orig, hypothesis_orig, target=None):
        import torch
        from torch.autograd import Variable
        """
        * **clsf** : **Classifier** .
        * **x_orig** : Input sentence.
        'entailment': 0, 'neutral': 1, 'contradiction': 2
        """
        # tokenization
        if self.lowercase:
            hypothesis_orig = hypothesis_orig.strip().lower()
            premise_orig = premise_orig.strip().lower()

        premise_words = premise_orig.strip().split(" ")
        hypothesis_words = hypothesis_orig.strip().split(" ")
        premise_words = ['<sos>'] + premise_words
        premise_words += ['<eos>']
        hypothesis_words = ['<sos>'] + hypothesis_words
        # hypothesis_words += ['<eos>']

        if ((len(premise_words) > self.maxlen + 1) or \
                (len(hypothesis_words) > self.maxlen)):
            print("Sentence too long!")
            return

        vocab = self.word2idx
        unk_idx = vocab['<oov>']
        hypothesis_indices = [vocab[w] if w in vocab else unk_idx for w in hypothesis_words]
        premise_indices = [vocab[w] if w in vocab else unk_idx for w in premise_words]
        premise_words = [w if w in vocab else '<oov>' for w in premise_words]
        hypothesis_words = [w if w in vocab else '<oov>' for w in hypothesis_words]
        length = min(len(hypothesis_words), self.maxlen)

        if len(premise_indices) < self.maxlen:
            premise_indices += [0] * (self.maxlen - len(premise_indices))
            premise_words += ["<pad>"] * (self.maxlen - len(premise_words))

        if len(hypothesis_indices) < self.maxlen:
            hypothesis_indices += [0] * (self.maxlen - len(hypothesis_indices))
            hypothesis_words += ["<pad>"] * (self.maxlen - len(hypothesis_words))

        premise = premise_indices[:self.maxlen]
        hypothesis = hypothesis_indices[:self.maxlen]
        premise_words = premise_words[:self.maxlen]
        hypothesis_words = hypothesis_words[:self.maxlen]
        # target = target = clsf.get_pred([x_orig])[0]
        if target is None:  # 0, 1, 2
            target = clsf.get_pred([premise_orig, hypothesis_orig])[0]  # calc x_orig's prediction
        # 这里应该怎么写？

        c = self.autoencoder.encode([hypothesis], [length], noise=False)  # 这里原本是一整个batch?
        z = self.inverter(c).data.cpu()

        # search_fast func
        # if not self.hybrid:

        premise = premise.unsqueeze(0)
        hypothesis = hypothesis.unsqueeze(0)
        y = target
        x_adv1, d_adv1 = None, None
        right_curr = self.right
        counter = 0

        while counter <= 5:
            mus = z.repeat(self.nsamples, 1)
            delta = torch.FloatTensor(mus.size()).uniform_(-1 * right_curr, right_curr)
            dist = np.array([np.sqrt(np.sum(x ** 2)) for x in delta.cpu().numpy()])
            perturb_z = Variable(mus + delta, volatile=True)
            x_tilde = self.generator(perturb_z)  # perturb

            y_tide = self.clsf([premise, hypothesis])[0]
            indices_adv = np.where(y_tide.data.cpu().numpy() != y.data.cpu().numpy())[0]

            if(len(indices_adv) > 0) and (indices_adv[0] == 0):
                indices_adv = np.delete(indices_adv, 0)

            if len(indices_adv) == 0:
                counter += 1
                right_curr *= 2
            else:
                idx_adv = get_min(indices_adv, dist)
                # if d_adv is None or (dist[idx_adv] < d_adv)
                x_adv1 = x_tilde[idx_adv]
                d_adv1 = float(dist[idx_adv])

        try:
            hyp_sample_idx = self.autoencoder.generate(x_adv1, 10, True).data.cpu().numpy()[0]
            words = [self.idx2word[x] for x in hyp_sample_idx]  # test
            if "<eos>" in words:
                words = words[:words.index("<eos>")]
            # target = clsf.get_pred([premise_orig, hypothesis_orig])[0]
            return(" ".join(words), clsf.get_pred([premise_orig, " ".join(words)])[0])

        except Exception as e:
            print(e)
            print(premise_words)
            print(hypothesis_words)
            print("no adversary found for : \n {0} \n {1}\n\n". \
                  format(" ".join(premise_words), " ".join(hypothesis_words)))
            return None

