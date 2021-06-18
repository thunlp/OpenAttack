import OpenAttack
import transformers
import datasets

def dataset_mapping(x):
    return {
        "x": x["sentence"],
        "y": 1 if x["label"] > 0.5 else 0,
    }
    
def main():
    print("Load model")
    tokenizer = transformers.AutoTokenizer.from_pretrained("./data/Victim.BERT.SST")
    model = transformers.AutoModelForSequenceClassification.from_pretrained("./data/Victim.BERT.SST", num_labels=2, output_hidden_states=False)
    clsf = OpenAttack.classifiers.TransformersClassifier (model, tokenizer, model.bert.embeddings.word_embeddings)

    print("New Attacker")
    attacker = OpenAttack.attackers.PWWSAttacker()

    dataset = datasets.load_dataset("sst", split="train[:20]").map(function=dataset_mapping)

    print("Start attack")
    attack_eval = OpenAttack.AttackEval(attacker, clsf, metrics = [
        OpenAttack.metric.EditDistance(),
        OpenAttack.metric.ModificationRate()
    ])
    attack_eval.eval(dataset, visualize=True)

if __name__ == "__main__":
    main()
