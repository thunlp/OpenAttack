class Jaccard_Word:
    def __call__(self, tokenA, tokenB):

        AS=set()
        BS=set()
        for i in range(len(tokenA)):
            AS.add(tokenA[i])
        for i in range(len(tokenB)):
            BS.add(tokenB[i])

        return len(AS&BS)/len(AS|BS)