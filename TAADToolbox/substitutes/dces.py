'''
    find the nearest neighbours in the list of unicode characters by finding characters
    with the largest number of matching tokens in the text description.
    根据unicode description找到描述相似的字符
    __call__函数参数：原字符，相似字符数量
    返回值：（字符，prob）列表，prob基本为均分，和为1
    进一步工作：对一段文本以概率p扰动
'''

from ..substitute import Substitute
# from ..exceptions import
import pandas as pd
import numpy as np
import random
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import CountVectorizer


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


class DcesSubstitute(Substitute):

    def __init__(self):
        self.descs = pd.read_csv('NamesList.txt', skiprows=np.arange(16), header=None, names=['code', 'description'],
                                 delimiter='\t')
        self.descs = self.descs.dropna(0)
        self.descs_arr = self.descs.values
        self.vectorizer = CountVectorizer(max_features=1000)
        self.desc_vecs = self.vectorizer.fit_transform(self.descs_arr[:, 0]).astype(float)
        self.vecsize = self.desc_vecs.shape[1]
        self.vec_colnames = np.arange(self.vecsize)
        self.desc_vecs = pd.DataFrame(self.desc_vecs.todense(), index=self.descs.index, columns=self.vec_colnames)
        self.descs = pd.concat([self.descs, self.desc_vecs], axis=1)
        # 预处理，准备好所有字符的描述

    def __call__(self, char, threshold):  # 原字符，topn

        c = get_hex_string(char)
        # 二进制编码

        if np.any(self.descs['code'] == c):
            description = self.descs['description'][self.descs['code'] == c].values[0]
        else:
            # raise exception
            print("failed to disturb %s" % char)
            return char, np.array([])
        # 找不到字符

        tokens = description.split(' ')  # 描述
        # print(tokens)
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
            print("cannot disturb, find no match")
            return [], []  # cannot disturb this one

        match_vecs = self.descs[self.vec_colnames].loc[matches]  # description

        neigh = NearestNeighbors(metric='euclidean')  # nearest neighbors
        Y = match_vecs.values
        neigh.fit(Y)

        X = self.descs[self.vec_colnames].values[self.descs['code'] == c]

        if Y.shape[0] > threshold:
            dists, idxs = neigh.kneighbors(X, threshold, return_distance=True)
        else:
            dists, idxs = neigh.kneighbors(X, Y.shape[0], return_distance=True)

        # turn distances to some heuristic probabilities
        # print(dists.flatten())
        probs = np.exp(-0.5 * dists.flatten())
        probs = probs / np.sum(probs)

        # turn idxs back to chars
        # print(idxs.flatten())
        charcodes = self.descs['code'][matches[idxs.flatten()]]

        # print(charcodes.values.flatten())

        chars = []
        for charcode in charcodes:
            chars.append(chr(int(charcode, 16)))
        print(chars)
        ret = []
        for i in range(len(chars)):
            ret.append((chars[i], probs[i]))  # 以概率
        # for char in chars:
        #    ret.append((char, 1))
        return ret

    # def disturb_sentence(self, sentence, prob):  # 以概率prob对一句话进行替换
