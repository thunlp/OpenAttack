import OpenAttack
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np
import datasets
import transformers

def make_model():
    class MyClassifier(OpenAttack.Classifier):
        def __init__(self):
            try:
                self.model = SentimentIntensityAnalyzer()
            except LookupError:
                nltk.download('vader_lexicon')
                self.model = SentimentIntensityAnalyzer()
        
        def get_pred(self, input_):
            return self.get_prob(input_).argmax(axis=1)

        def get_prob(self, input_):
            ret = []
            for sent in input_:
                res = self.model.polarity_scores(sent)
                prob = (res["pos"] + 1e-6) / (res["neg"] + res["pos"] + 1e-6)
                ret.append(np.array([1 - prob, prob]))
            return np.array(ret)
    return MyClassifier()

def dataset_mapping(x):
    return {
        "x": x["sentence"],
        "y": 1 if x["label"] > 0.5 else 0,
    }

def main():
    print("New Attacker")
    #attacker = OpenAttack.attackers.PWWSAttacker()
    print("Build model")
    #clsf = OpenAttack.loadVictim("BERT.SST")
    tokenizer = transformers.AutoTokenizer.from_pretrained("./data/Victim.BERT.SST")
    model = transformers.AutoModelForSequenceClassification.from_pretrained("./data/Victim.BERT.SST", num_labels=2, output_hidden_states=True)
    clsf = OpenAttack.classifiers.TransformersClassifier(model, tokenizer=tokenizer, max_length=100, embedding_layer=model.bert.embeddings.word_embeddings)

    dataset = datasets.load_dataset("sst", split="train[:5]").map(function=dataset_mapping)
    print("New Attacker")
    attacker = OpenAttack.attackers.GEOAttacker(data=dataset)
    print("Start attack")
    attack_eval = OpenAttack.AttackEval( attacker, clsf, metrics=[
        OpenAttack.metric.Fluency(),
        OpenAttack.metric.GrammaticalErrors(),
        OpenAttack.metric.SemanticSimilarity(),
        OpenAttack.metric.EditDistance(),
        OpenAttack.metric.ModificationRate()
    ] )
    attack_eval.eval(dataset, visualize=True, progress_bar=True)

if __name__ == "__main__":
    main()
