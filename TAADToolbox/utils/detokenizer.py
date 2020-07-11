def detokenizer(tokens):
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