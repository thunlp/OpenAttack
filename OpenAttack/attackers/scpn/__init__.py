import os
import pickle
import numpy as np
from ...utils import check_parameters
from ...text_processors import DefaultTextProcessor
from ...attacker import Attacker
from ...data_manager import DataManager

DEFAULT_TEMPLATES = [
    '( ROOT ( S ( NP ) ( VP ) ( . ) ) ) EOP',
    '( ROOT ( S ( VP ) ( . ) ) ) EOP',
    '( ROOT ( NP ( NP ) ( . ) ) ) EOP',
    '( ROOT ( FRAG ( SBAR ) ( . ) ) ) EOP',
    '( ROOT ( S ( S ) ( , ) ( CC ) ( S ) ( . ) ) ) EOP',
    '( ROOT ( S ( LST ) ( VP ) ( . ) ) ) EOP',
    '( ROOT ( SBARQ ( WHADVP ) ( SQ ) ( . ) ) ) EOP',
    '( ROOT ( S ( PP ) ( , ) ( NP ) ( VP ) ( . ) ) ) EOP',
    '( ROOT ( S ( ADVP ) ( NP ) ( VP ) ( . ) ) ) EOP',
    '( ROOT ( S ( SBAR ) ( , ) ( NP ) ( VP ) ( . ) ) ) EOP'
]

DEFAULT_CONFIG = {
    "templates": DEFAULT_TEMPLATES,
    "device": None,
    "processor": DefaultTextProcessor()
}


def reverse_bpe(sent):
    x = []
    cache = ''
    for w in sent:
        if w.endswith('@@'):
            cache += w.replace('@@', '')
        elif cache != '':
            x.append(cache + w)
            cache = ''
        else:
            x.append(w)
    return ' '.join(x)


class SCPNAttacker(Attacker):
    def __init__(self, **kwargs):
        """
        :param list templates: A list of templates used in SCPNAttacker. **Default:** ten manually selected templates.
        :param torch.device device: The device to load SCPN models (pytorch). **Default:** Use "cpu" if cuda is not available else "cuda".
        :param TextProcessor processor: Text processor used in this attacker. **Default:** :any:`DefaultTextProcessor`.
        
        :Package Requirements:
            * torch
        :Data Requirements: :py:data:`.AttackAssist.SCPN`
        :Classifier Capacity: Blind

        Adversarial Example Generation with Syntactically Controlled Paraphrase Networks. Mohit Iyyer, John Wieting, Kevin Gimpel, Luke Zettlemoyer. NAACL-HLT 2018.
        `[pdf] <https://www.aclweb.org/anthology/N18-1170.pdf>`__
        `[code] <https://github.com/miyyer/scpn>`__

        The default templates are:
        
        .. code-block:: python
           
            DEFAULT_TEMPLATES = [
                '( ROOT ( S ( NP ) ( VP ) ( . ) ) ) EOP',
                '( ROOT ( S ( VP ) ( . ) ) ) EOP',
                '( ROOT ( NP ( NP ) ( . ) ) ) EOP',
                '( ROOT ( FRAG ( SBAR ) ( . ) ) ) EOP',
                '( ROOT ( S ( S ) ( , ) ( CC ) ( S ) ( . ) ) ) EOP',
                '( ROOT ( S ( LST ) ( VP ) ( . ) ) ) EOP',
                '( ROOT ( SBARQ ( WHADVP ) ( SQ ) ( . ) ) ) EOP',
                '( ROOT ( S ( PP ) ( , ) ( NP ) ( VP ) ( . ) ) ) EOP',
                '( ROOT ( S ( ADVP ) ( NP ) ( VP ) ( . ) ) ) EOP',
                '( ROOT ( S ( SBAR ) ( , ) ( NP ) ( VP ) ( . ) ) ) EOP'
            ]
        
        """
        self.models = __import__("models", globals={
            "__name__":__name__,
            "__package__": __package__,
        }, level=1)
        self.subword = __import__("subword", globals={
            "__name__":__name__,
            "__package__": __package__,
        }, level=1)
        self.torch = __import__("torch")

        self.config = DEFAULT_CONFIG.copy()
        self.config.update(kwargs)
        check_parameters(DEFAULT_CONFIG, self.config)
        
        if self.config["device"] is None:
            if self.torch.cuda.is_available():
                self.device = self.torch.device("cuda")
            else:
                self.device = self.torch.device("cpu")
        else:
            self.device = self.torch.device( self.config["device"] )
        
        self.processor = self.config["processor"]

        # Use DataManager Here
        model_path = DataManager.load("AttackAssist.SCPN")
        pp_model = self.torch.load(model_path["scpn.pt"], map_location=self.device)
        parse_model = self.torch.load(model_path["parse_generator.pt"], map_location=self.device)
        pp_vocab, rev_pp_vocab = pickle.load(open(model_path["parse_vocab.pkl"], 'rb'))
        bpe_codes = open(model_path["bpe.codes"], "r", encoding="utf-8")
        bpe_vocab = open(model_path["vocab.txt"], "r", encoding="utf-8")
        self.parse_gen_voc = pickle.load(open(model_path["ptb_tagset.pkl"], "rb"))


        self.pp_vocab = pp_vocab
        self.rev_pp_vocab = rev_pp_vocab
        self.rev_label_voc = dict((v,k) for (k,v) in self.parse_gen_voc.items())

        # load paraphrase network
        pp_args = pp_model['config_args']
        self.net = self.models.SCPN(pp_args.d_word, pp_args.d_hid, pp_args.d_nt, pp_args.d_trans, len(self.pp_vocab), len(self.parse_gen_voc) - 1, pp_args.use_input_parse)
        self.net.load_state_dict(pp_model['state_dict'])
        self.net = self.net.to(self.device).eval()

        # load parse generator network
        parse_args = parse_model['config_args']
        self.parse_net = self.models.ParseNet(parse_args.d_nt, parse_args.d_hid, len(self.parse_gen_voc))
        self.parse_net.load_state_dict(parse_model['state_dict'])
        self.parse_net = self.parse_net.to(self.device).eval()

        # instantiate BPE segmenter
        
        bpe_vocab = self.subword.read_vocabulary(bpe_vocab, 50)
        self.bpe = self.subword.BPE(bpe_codes, '@@', bpe_vocab, None)

    def gen_paraphrase(self, sent, templates):
        template_lens = [len(x.split()) for x in templates]
        np_templates = np.zeros((len(templates), max(template_lens)), dtype='int32')
        for z, template in enumerate(templates):
            np_templates[z, :template_lens[z]] = [self.parse_gen_voc[w] for w in templates[z].split()]
        tp_templates = self.torch.from_numpy(np_templates).long().to(self.device)
        tp_template_lens = self.torch.LongTensor(template_lens).to(self.device)

        ssent = ' '.join(list(map(lambda x:x[0], self.processor.get_tokens(sent))))
        seg_sent = self.bpe.segment(ssent.lower()).split()
        
        # encode sentence using pp_vocab, leave one word for EOS
        seg_sent = [self.pp_vocab[w] for w in seg_sent if w in self.pp_vocab]

        # add EOS
        seg_sent.append(self.pp_vocab['EOS'])
        torch_sent = self.torch.LongTensor(seg_sent).to(self.device)
        torch_sent_len = self.torch.LongTensor([len(seg_sent)]).to(self.device)

        # encode parse using parse vocab
        # Stanford Parser
        parse_tree = self.processor.get_parser(sent)
        parse_tree = " ".join(parse_tree.replace("\n", " ").split()).replace("(", "( ").replace(")", " )")
        parse_tree = parse_tree.split()

        for i in range(len(parse_tree) - 1):
            if (parse_tree[i] not in "()") and (parse_tree[i + 1] not in "()"):
                parse_tree[i + 1] = ""
        parse_tree = " ".join(parse_tree).split() + ["EOP"]

        torch_parse = self.torch.LongTensor([self.parse_gen_voc[w] for w in parse_tree]).to(self.device)
        torch_parse_len = self.torch.LongTensor([len(parse_tree)]).to(self.device)

        # generate full parses from templates
        beam_dict = self.parse_net.batch_beam_search(torch_parse.unsqueeze(0), tp_templates, torch_parse_len[:], tp_template_lens, self.parse_gen_voc['EOP'], beam_size=3, max_steps=150)
        seq_lens = []
        seqs = []
        for b_idx in beam_dict:
            prob,_,_,seq = beam_dict[b_idx][0]
            seq = seq[:-1] # chop off EOP
            seq_lens.append(len(seq))
            seqs.append(seq)
        np_parses = np.zeros((len(seqs), max(seq_lens)), dtype='int32')
        for z, seq in enumerate(seqs):
            np_parses[z, :seq_lens[z]] = seq
        tp_parses = self.torch.from_numpy(np_parses).long().to(self.device)
        tp_len = self.torch.LongTensor(seq_lens).to(self.device)

        # generate paraphrases from parses
        ret = []
        beam_dict = self.net.batch_beam_search(torch_sent.unsqueeze(0), tp_parses, torch_sent_len[:], tp_len, self.pp_vocab['EOS'], beam_size=3, max_steps=40)
        for b_idx in beam_dict:
            prob,_,_,seq = beam_dict[b_idx][0]
            gen_parse = ' '.join([self.rev_label_voc[z] for z in seqs[b_idx]])
            gen_sent = ' '.join([self.rev_pp_vocab[w] for w in seq[:-1]])
            ret.append(reverse_bpe(gen_sent.split()))
        return ret
    def __call__(self, clsf, sent, target=None):
        if target is None:
            targeted = False
            target = clsf.get_pred([sent])[0]  # calc x_orig's prediction
        else:
            targeted = True

        try:
            pps = self.gen_paraphrase(sent, self.config["templates"])
        except KeyError as e:
            return None
        preds = clsf.get_pred(pps)

        if targeted:
            idx = (preds == target).argmax()
            if preds[idx] == target:
                return pps[idx], target
        else:
            idx = (preds == target).argmin()
            if preds[idx] != target:
                return pps[idx], preds[idx]
        return None
        
