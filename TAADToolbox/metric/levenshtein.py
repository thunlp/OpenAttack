import numpy as np
def levenshtein(a, b):
    la = len(a)
    lb = len(b)
    f = np.zeros((la + 1, lb + 1), dtype=np.uint64)
    for i in range(la + 1):
        for j in range(lb + 1):
            if i == 0:
                f[i][j] = j
            elif j == 0:
                f[i][j] = i
            elif a[i - 1] == b[j - 1]:
                f[i][j] = f[i - 1][j - 1]
            else:
                f[i][j] = min(f[i - 1][j - 1], f[i - 1][j], f[i][j - 1]) + 1
    return f[la][lb]