"""
    find the nearest neighbours in the list of unicode characters by finding characters
    with the largest number of matching tokens in the text description.
    根据unicode description找到描述相似的字符
    __call__函数参数：原字符，相似字符数量
    返回值：（字符，prob）列表，prob基本为均分，和为1
    require:
    DataManager.download("DCES")
"""

from .base import CharSubstitute
from ..data_manager import DataManager
# from ..exceptions import
import numpy as np


disallowed = ['TAG', 'MALAYALAM', 'BAMUM', 'HIRAGANA', 'RUNIC', 'TAI', 'SUNDANESE', 'BATAK', 'LEPCHA', 'CHAM',
              'TELUGU', 'DEVANGARAI', 'BUGINESE', 'MYANMAR', 'LINEAR', 'SYLOTI', 'PHAGS-PA', 'CHEROKEE',
              'CANADIAN', 'YI', 'LYCIAN', 'HANGUL', 'KATAKANA', 'JAVANESE', 'ARABIC', 'KANNADA', 'BUHID',
              'TAGBANWA', 'DESERET', 'REJANG', 'BOPOMOFO', 'PERMIC', 'OSAGE', 'TAGALOG', 'MEETEI', 'CARIAN',
              'UGARITIC', 'ORIYA', 'ELBASAN', 'CYPRIOT', 'HANUNOO', 'GUJARATI', 'LYDIAN', 'MONGOLIAN', 'AVESTAN',
              'MEROITIC', 'KHAROSHTHI', 'HUNGARIAN', 'KHUDAWADI', 'ETHIOPIC', 'PERSIAN', 'OSMANYA', 'ELBASAN',
              'TIBETAN', 'BENGALI', 'TURKIC', 'THROWING', 'HANIFI', 'BRAHMI', 'KAITHI', 'LIMBU', 'LAO', 'CHAKMA',
              'DEVANAGARI', 'ITALIC', 'CJK', 'MEDEFAIDRIN', 'DIAMOND', 'SAURASHTRA', 'ADLAM', 'DUPLOYAN']
disallowed_codes = ['1F1A4', 'A7AF']  # 不允许编码


def get_hex_string(ch):
    return '{:04x}'.format(ord(ch)).upper()  # 获得字符16进制编码


class DcesSubstitute(CharSubstitute):
    """
    :Data Requirements: :any:`DCES`
    
    An implementation of :py:class:`.CharSubstitute`.

    DCES substitute used in :py:class:`.VIPERAttacker`.

    """

    def __init__(self):
        self.vec_colnames, self.descs, self.neigh = DataManager.load("DCES")
        # load

    def __call__(self, char, threshold):  # 原字符，topn
        c = get_hex_string(char)
        # 二进制编码

        if np.any(self.descs['code'] == c):
            description = self.descs['description'][self.descs['code'] == c].values[0]
        else:
            # raise exception
            # print("failed to disturb %s" % char)
            return [char, 1]
        # 找不到字符

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
        for i in identifiers:
            for idx in self.descs.index:
                desc_toks = self.descs['description'][idx].split(' ')
                if i in desc_toks and not np.any(np.in1d(desc_toks, disallowed)) and \
                        not np.any(np.in1d(self.descs['code'][idx], disallowed_codes)) and \
                        not int(self.descs['code'][idx], 16) > 30000:

                    # get the first case descriptor in the description
                    desc_toks = np.array(desc_toks)
                    case_descriptor = desc_toks[(desc_toks == 'SMALL') | (desc_toks == 'CAPITAL')]

                    if len(case_descriptor) > 1:
                        case_descriptor = case_descriptor[0]
                    elif len(case_descriptor) == 0:
                        case = 'unknown'

                    if case == 'unknown' or case == case_descriptor:
                        matches.append(idx)

        # return c, np.array(matches)
        matches = np.array(matches)  # 找到所有共同token的字符

        if not len(matches):
            # print("cannot disturb, find no match")
            return [(char, 1)]  # cannot disturb this one

        match_vecs = self.descs[self.vec_colnames].loc[matches]  # description

        Y = match_vecs.values
        self.neigh.fit(Y)

        X = self.descs[self.vec_colnames].values[self.descs['code'] == c]

        if Y.shape[0] > threshold:
            dists, idxs = self.neigh.kneighbors(X, threshold, return_distance=True)
        else:
            dists, idxs = self.neigh.kneighbors(X, Y.shape[0], return_distance=True)

        # turn distances to some heuristic probabilities
        probs = np.exp(-0.5 * dists.flatten())
        probs = probs / np.sum(probs)

        # turn idxs back to chars
        # print(idxs.flatten())
        charcodes = self.descs['code'][matches[idxs.flatten()]]

        chars = []
        for charcode in charcodes:
            chars.append(chr(int(charcode, 16)))
        # print(chars)
        ret = []
        for i in range(len(chars)):
            ret.append((chars[i], probs[i]))  # 以概率
        # for char in chars:
        #    ret.append((char, 1))
        return ret
