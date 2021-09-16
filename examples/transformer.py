'''
This example code illustrates adversarial attacks against a fine-tuned model from Transformers on a dataset from Datasets.
'''
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
    tokenizer = transformers.AutoTokenizer.from_pretrained("echarlaix/bert-base-uncased-sst2-acc91.1-d37-hybrid")
    model = transformers.AutoModelForSequenceClassification.from_pretrained("echarlaix/bert-base-uncased-sst2-acc91.1-d37-hybrid", num_labels=2, output_hidden_states=False)
    victim = OpenAttack.classifiers.TransformersClassifier (model, tokenizer, model.bert.embeddings.word_embeddings)

    print("New Attacker")
    attacker = OpenAttack.attackers.PWWSAttacker()

    dataset = datasets.load_dataset("sst", split="train[:20]").map(function=dataset_mapping)

    print("Start attack")
    attack_eval = OpenAttack.AttackEval(attacker, victim, metrics = [
        OpenAttack.metric.EditDistance(),
        OpenAttack.metric.ModificationRate()
    ])
    attack_eval.eval(dataset, visualize=True)

if __name__ == "__main__":
    main()
