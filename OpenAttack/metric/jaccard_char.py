class Jaccard_Char:
    def __call__(self, senA, senB):
        """
            :param list tokenA: The first list of tokens.
            :param list tokenB: The second list of tokens.

            Make sure two list have the same length.
        """
        AS=set()
        BS=set()
        for i in range(len(senA)):
            AS.add(senA[i])
        for i in range(len(senB)):
            BS.add(senB[i])

        return len(AS&BS)/len(AS|BS)