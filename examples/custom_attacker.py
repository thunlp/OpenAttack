import TAADToolbox as tat

class MyAttacker(tat.Attacker):
    def __init__(self, max_iter=20, processor = tat.text_processors.DefaultTextProcessor()):
        self.processor = processor
        self.max_iter = max_iter
    
    def __call__(self, clsf, x_orig, target=None):
        if target is None:
            target = clsf.get_pred([x_orig])[0]
            targeted = False
        else:
            targeted = True
        
        # generate samples
        all_sents = []
        curr_x = self.processor.get_tokens(x_orig)
        for i in range(self.max_iter):
            curr_x = self.swap(curr_x)
            sent = tat.utils.detokenizer(curr_x)
            all_sents.append(sent)
        
        # get prediction
        preds = clsf.get_pred(all_sents)

        for idx, sent in enumerate(all_sents):
            if targeted:
                if preds[idx] == target:
                    return (sent, preds[idx])
            else:
                if preds[idx] != target:
                    return (sent, preds[idx])
        return None
    
    def swap(self, sent_token):
        pairs = []
        for i in range(len(sent_token)):
            for j in range(i):
                if sent_token[i][1] == sent_token[j][1]:    # same POS
                    pairs.append((i, j))
        if len(pairs) == 0:
            return sent_token

        import random
        pi, pj = random.choice(pairs)   # random select one pair
        sent_token[pi], sent_token[pj] = sent_token[pj], sent_token[pi] # swap this pair
        return sent_token


def main():
    word_vector = tat.DataManager.load("Glove")
    model = tat.DataManager.load("Victim.BiLSTM.SST")
    dataset = tat.DataManager.load("Dataset.SST.sample")[:10]

    clsf = tat.classifiers.PytorchClassifier(model, 
                word2id=word_vector.word2id, embedding=word_vector.get_vecmatrix(), 
                token_unk= "UNK", require_length=True, device="cpu")
    attacker = MyAttacker()
    attacker_eval = tat.attacker_evals.DefaultAttackerEval(attacker, clsf)
    print( attacker_eval.eval(dataset) )

if __name__ == "__main__":
    main()