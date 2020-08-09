import os
import numpy as np
import nltk

tokenizer = nltk.WordPunctTokenizer().tokenize
def right_bar_print(info, key_len=20, val_len=10):
    ret = []
    ret.append( " " * (key_len + val_len) )
    for key, val in info.items():
        row = " %s: " % (key[:key_len - 3])
        row += " " * (key_len - len(row))
        if isinstance(val, bool):
            if val:
                row += " yes" + " " * (val_len - 4)
            else:
                row += " no" + " " * (val_len - 3)
        elif isinstance(val, int):
            val_str = " %d" % val
            row += val_str + " " * (val_len - len(val_str))
        elif isinstance(val, float):
            val_str = " %.5g" % val
            if len(val_str) > val_len:
                val_str = (" %.7f" % val)[:val_len]
            row += val_str + " " * (val_len - len(val_str))
        else:
            val_str = (" %s" % val)[:val_len]
            row += val_str + " " * (val_len - len(val_str))
        ret.append(row)
    ret.append( " " * (key_len + val_len) )
    return ret

def word_align(wordA, wordB):
    if len(wordA) < len(wordB):
        wordA += " " * (len(wordB) - len(wordA))
    else:
        wordB += " " * (len(wordA) - len(wordB))
    return wordA, wordB

def levenshtein_visual(a, b):
    la = len(a)
    lb = len(b)
    f = np.zeros((la + 1, lb + 1), dtype=np.uint64)
    for i in range(la + 1):
        for j in range(lb + 1):
            if i == 0:
                f[i][j] = j
            elif j == 0:
                f[i][j] = i
            elif a[i - 1].lower() == b[j - 1].lower():
                f[i][j] = f[i - 1][j - 1]
            else:
                f[i][j] = min(f[i - 1][j - 1], f[i - 1][j], f[i][j - 1]) + 1
    
    p, q = la, lb
    ret = []
    while p > 0 and q > 0:
        if a[p - 1].lower() == b[q - 1].lower():
            ret.append( (a[p - 1], b[q - 1]) )
            p -= 1
            q -= 1
        else:
            if f[p][q] == f[p - 1][q - 1] + 1:
                # modify
                ret.append( word_align(a[p - 1], b[q - 1]) )
                p -= 1
                q -= 1
            elif f[p][q] == f[p - 1][q] + 1:
                # remove
                ret.append( word_align(a[p - 1], "") )
                p -= 1
            else:
                assert f[p][q] == f[p][q - 1] + 1
                ret.append( word_align("", b[q - 1]) )
                q -= 1
    while p > 0:
        ret.append( word_align( a[p - 1], "" ) )
        p -= 1
    while q > 0:
        ret.append( word_align( "", b[q - 1] ) )
        q -= 1
    return ret[::-1]

def left_bar_print(x_orig, y_orig, x_adv, y_adv, max_len):
    ret = []

    assert isinstance(y_orig, int) == isinstance(y_adv, int)
    if isinstance(y_orig, int):
        head_str = "Label: %d --> %d" % (y_orig, y_adv)
    else:
        head_str = "Label: %d (%.2lf%%) --> %d (%.2lf%%)" % (y_orig.argmax(), y_orig.max() * 100, y_adv.argmax(), y_adv.max() * 100)
    ret.append(("\033[32m%s\033[0m" % head_str) + " " * (max_len - len(head_str)))
    ret.append(" " * max_len)
    
    token_orig = tokenizer(x_orig)
    token_adv = tokenizer(x_adv)
    pairs = levenshtein_visual(token_orig, token_adv)
    
    curr1 = ""
    curr2 = ""
    length = 0
    for tokenA, tokenB in pairs:
        assert len(tokenA) == len(tokenB)
        if length + len(tokenA) + 1 > max_len:
            ret.append(curr1 + " " * (max_len - length))
            ret.append(curr2 + " " * (max_len - length))
            ret.append(" " * max_len)
            length = len(tokenA) + 1
            if tokenA.lower() == tokenB.lower():
                curr1 = tokenA + " "
                curr2 = tokenB + " "
            else:
                curr1 = "\033[1;31m" + tokenA + "\033[0m" + " "
                curr2 = "\033[1;32m" + tokenB + "\033[0m" + " "
        else:
            length += 1 + len(tokenA)
            if tokenA.lower() == tokenB.lower():
                curr1 += tokenA + " "
                curr2 += tokenB + " "
            else:
                curr1 += "\033[1;31m" + tokenA + "\033[0m" + " "
                curr2 += "\033[1;32m" + tokenB + "\033[0m" + " "
    if length > 0:
        ret.append(curr1 + " " * (max_len - length))
        ret.append(curr2 + " " * (max_len - length))
        ret.append(" " * max_len)
    return ret

def left_bar_failed(x_orig, y_orig, max_len):
    ret = []

    if isinstance(y_orig, int):
        head_str = "Label: %d --> Failed!" % y_orig
    else:
        head_str = "Label: %d (%.2lf%%) --> Failed!" % (y_orig.argmax(), y_orig.max() * 100)
    ret.append(("\033[31m%s\033[0m" % head_str) + " " * (max_len - len(head_str)))
    ret.append(" " * max_len)
    tokens = x_orig.split()
    curr = ""
    for tk in tokens:
        if len(curr) + len(tk) + 1 > max_len:
            ret.append(curr + " " * (max_len - len(curr)))
            curr = tk + " "
        else:
            curr += tk + " "
    if len(curr) > 0:
        ret.append(curr + " " * (max_len - len(curr)))
    ret.append(" " * max_len)
    return ret

def visualizer(idx, x_orig, y_orig, x_adv, y_adv, info, stream_writer, key_len=25, val_len=10):
    """
    Visualization tools used in :py:class:`.DefaultAttackEval`.
    """
    try:
        cols = os.get_terminal_size().columns
    except OSError:
        cols = 80

    headline = "Sample: %d " % idx
    headline = headline + ("=" * (cols - len(headline) - 1)) + "\n"
    stream_writer(headline)

    max_len = cols - 1 - key_len - val_len

    right = right_bar_print(info, key_len=key_len, val_len=val_len)
    if x_adv is None:
        # Failed
        left = left_bar_failed(x_orig, y_orig, max_len)
    else:
        left = left_bar_print(x_orig, y_orig, x_adv, y_adv, max_len)
    
    if len(left) < len(right):
        delta = len(right) - len(left)
        if delta % 2 == 1:
            left.append(" " * max_len)
            delta -= 1
        while delta > 0:
            delta -= 2
            left.insert(1, " " * max_len)
            left.append(" " * max_len)
    elif len(right) < len(left):
        delta = len(left) - len(right)
        if delta % 2 == 1:
            right.append(" " * (key_len + val_len))
            delta -= 1
        while delta > 0:
            delta -= 2
            right.insert(0, " " * (key_len + val_len))
            right.append(" " * (key_len + val_len))
    assert len(left) == len(right)
    for l, r in zip(left, right):
        stream_writer(l)
        stream_writer("|")
        stream_writer(r)
        stream_writer("\n")
    

def result_visualizer(result, stream_writer):
    """
    Visualization tools used in :py:class:`.DefaultAttackEval`.
    """
    try:
        cols = os.get_terminal_size().columns
    except OSError:
        cols = 80

    left = []
    right = []
    for key, val in result.items():
        left.append(" " + key + ": ")
        if isinstance(val, bool):
            right.append(" " + "yes" if val else "no" )
        elif isinstance(val, int):
            right.append(" %d" % val)
        elif isinstance(val, float):
            right.append(" %.5g" % val)
        else:
            right.append(" %s" % val)
        right[-1] += " "
    
    max_left = max(list(map(len, left)))
    max_right = max(list(map(len, right)))
    if max_left + max_right + 3 > cols:
        delta = max_left + max_right + 3 - cols
        if delta % 2 == 1:
            delta -= 1
            max_left -= 1
        max_left -= delta // 2
        max_right -= delta // 2
    total = max_left + max_right + 3

    title = "Summary"
    if total - 2 < len(title):
        title = title[:total - 2]
    offtitle = ((total - len(title)) // 2) - 1
    stream_writer("+" + ("=" * (total - 2)) + "+\n" )
    stream_writer("|" + " " * offtitle + title + " " * (total - 2 - offtitle - len(title)) + "|" + "\n")
    stream_writer("+" + ("=" * (total - 2)) + "+\n" )
    for l, r in zip(left, right):
        l = l[:max_left]
        r = r[:max_right]
        l += " " * (max_left - len(l))
        r += " " * (max_right - len(r))
        stream_writer("|" + l + "|" + r + "|" + "\n")
    stream_writer("+" + ("=" * (total - 2)) + "+\n" )


