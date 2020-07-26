class Modification:
    def __call__(self, tokenA, tokenB):
        """
            :param list tokenA: The first list of tokens.
            :param list tokenB: The second list of tokens.

            Make sure two list have the same length.
        """
        va = tokenA
        vb = tokenB
        ret = 0
        if len(va) != len(vb):
            ret = abs(len(va) - len(vb))
        mn_len = min(len(va), len(vb))
        va, vb = va[:mn_len], vb[:mn_len]
        for wordA, wordB in zip(va, vb):
            if wordA != wordB:
                ret += 1
        return ret / len(va)
