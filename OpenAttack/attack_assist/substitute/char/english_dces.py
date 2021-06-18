from .base import CharSubstitute
from ....data_manager import DataManager
from ....tags import *
import numpy as np


disallowed = ['TAG', 'MALAYALAM', 'BAMUM', 'HIRAGANA', 'RUNIC', 'TAI', 'SUNDANESE', 'BATAK', 'LEPCHA', 'CHAM',
              'TELUGU', 'DEVANGARAI', 'BUGINESE', 'MYANMAR', 'LINEAR', 'SYLOTI', 'PHAGS-PA', 'CHEROKEE',
              'CANADIAN', 'YI', 'LYCIAN', 'HANGUL', 'KATAKANA', 'JAVANESE', 'ARABIC', 'KANNADA', 'BUHID',
              'TAGBANWA', 'DESERET', 'REJANG', 'BOPOMOFO', 'PERMIC', 'OSAGE', 'TAGALOG', 'MEETEI', 'CARIAN',
              'UGARITIC', 'ORIYA', 'ELBASAN', 'CYPRIOT', 'HANUNOO', 'GUJARATI', 'LYDIAN', 'MONGOLIAN', 'AVESTAN',
              'MEROITIC', 'KHAROSHTHI', 'HUNGARIAN', 'KHUDAWADI', 'ETHIOPIC', 'PERSIAN', 'OSMANYA', 'ELBASAN',
              'TIBETAN', 'BENGALI', 'TURKIC', 'THROWING', 'HANIFI', 'BRAHMI', 'KAITHI', 'LIMBU', 'LAO', 'CHAKMA',
              'DEVANAGARI', 'ITALIC', 'CJK', 'MEDEFAIDRIN', 'DIAMOND', 'SAURASHTRA', 'ADLAM', 'DUPLOYAN']
disallowed_codes = ['1F1A4', 'A7AF']  # filtered codes


def get_hex_string(ch):
    return '{:04x}'.format(ord(ch)).upper()  # Get the hex code of char


class DCESSubstitute(CharSubstitute):
    TAGS = { TAG_English }

    def __init__(self, k : int = 12):
        """
        Returns the chars that is visually similar to the input.

        DCES substitute used in :py:class:`.VIPERAttacker`.

        Args:
            k: Top-k results to return. Default: k = 12
        
        :Data Requirements: :py:data:`.AttackAssist.SIM`
        :Language: english
        :Package Requirements: * **sklearn**

        """
        self.descs, self.neigh = DataManager.load("AttackAssist.DCES")
        self.k = k

    def substitute(self, char: str):
        c = get_hex_string(char)

        if c in self.descs:
            description = self.descs[c]["description"]
        else:
            return [(char, 1)]

        tokens = description.split(' ')
        case = 'unknown'
        identifiers = []

        for token in tokens:
            if len(token) == 1:
                identifiers.append(token)
            elif token == 'SMALL':
                case = 'SMALL'
            elif token == 'CAPITAL':
                case = 'CAPITAL'

        matches = []
        match_ids = []
        for i in identifiers:
            for idx, val in self.descs.items():
                desc_toks = val["description"].split(' ')
                if i in desc_toks and not np.any(np.in1d(desc_toks, disallowed)) and \
                        not np.any(np.in1d(idx, disallowed_codes)) and \
                        not int(idx, 16) > 30000:

                    desc_toks = np.array(desc_toks)
                    case_descriptor = desc_toks[(desc_toks == 'SMALL') | (desc_toks == 'CAPITAL')]

                    if len(case_descriptor) > 1:
                        case_descriptor = case_descriptor[0]
                    elif len(case_descriptor) == 0:
                        case = 'unknown'

                    if case == 'unknown' or case == case_descriptor:
                        match_ids.append(idx)
                        matches.append(val["vec"])

        if len(matches) == 0:
            return [(char, 1)]

        match_vecs = np.stack(matches)
        Y = match_vecs

        self.neigh.fit(Y)

        X = self.descs[c]["vec"].reshape(1, -1)

        if Y.shape[0] > self.k:
            dists, idxs = self.neigh.kneighbors(X, self.k, return_distance=True)
        else:
            dists, idxs = self.neigh.kneighbors(X, Y.shape[0], return_distance=True)
        probs = dists.flatten()

        charcodes = [match_ids[idx] for idx in idxs.flatten()]
        
        chars = []
        for charcode in charcodes:
            chars.append(chr(int(charcode, 16)))
        ret = list(zip(chars, probs))
        return ret
