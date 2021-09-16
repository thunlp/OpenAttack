'''
This example code shows how to conduct adversarial attacks against a sentence pair classification (NLI) model
'''
import OpenAttack
import transformers
import datasets

class NLIWrapper(OpenAttack.classifiers.Classifier):
    def __init__(self, model : OpenAttack.classifiers.Classifier):
        self.model = model
    
    def get_pred(self, input_):
        return self.get_prob(input_).argmax(axis=1)

    def get_prob(self, input_):
        ref = self.context.input["hypothesis"]
        input_sents = [  sent + "</s></s>" + ref for sent in input_ ]
        print(input_sents)
        return self.model.get_prob(
            input_sents
        )


def dataset_mapping(x):
    return {
        "x": x["premise"],
        "y": x["label"],
        "hypothesis": x["hypothesis"]
    }
    
def main():
    print("Load model")
    tokenizer = transformers.AutoTokenizer.from_pretrained("roberta-large-mnli")
    model = transformers.AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli", output_hidden_states=False)
    victim = OpenAttack.classifiers.TransformersClassifier(model, tokenizer, model.roberta.embeddings.word_embeddings)
    victim = NLIWrapper(victim)

    print("New Attacker")
    attacker = OpenAttack.attackers.PWWSAttacker()

    dataset = datasets.load_dataset("glue", "mnli", split="train[:20]").map(function=dataset_mapping)

    print("Start attack")
    attack_eval = OpenAttack.AttackEval(attacker, victim, metrics = [
        OpenAttack.metric.EditDistance(),
        OpenAttack.metric.ModificationRate()
    ])
    attack_eval.eval(dataset, visualize=True)

if __name__ == "__main__":
    main()