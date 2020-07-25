def detokenizer(tokens):
    """
    :param list tokens: A list of tokens
    :return: A detokenized sentence.
    :rtype: str

    Detokenizer used in attackers. This function will be moved to `TextProcessor` in the future.
    """
    all_tuple = True
    for it in tokens:
        if not isinstance(it, tuple):
            all_tuple = False
    if all_tuple:
        tokens = list(map(lambda x:x[0], tokens))
        
    ret = ""
    new_sent = True
    for token in tokens:
        if token in ".?!":
            ret += token
            new_sent = True
        elif len(token) >= 2 and token[0] == "'" and token[1] != "'":
            ret += token
        elif len(token) >= 2 and token[:2] == "##":
            ret += token[2:]
        elif token == "n't":
            ret += token
        else:
            if new_sent:
                ret += " " + token.capitalize()
                new_sent = False
            else:
                ret += " " + token
    return ret