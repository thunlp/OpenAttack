import OpenAttack
import transformers
import datasets
    
def main():
    print("Load model")
    tokenizer = transformers.AutoTokenizer.from_pretrained("echarlaix/bert-base-uncased-sst2-acc91.1-d37-hybrid")
    model = transformers.AutoModelForSequenceClassification.from_pretrained("echarlaix/bert-base-uncased-sst2-acc91.1-d37-hybrid", num_labels=2, output_hidden_states=False)
    clsf = OpenAttack.classifiers.TransformersClassifier (model, tokenizer, model.bert.embeddings.word_embeddings)

    print("New Attacker")
    attacker = OpenAttack.attackers.PWWSAttacker()

    # create your dataset here
    dataset = datasets.Dataset.from_dict({
        "x": [
            "I hate this movie.",
            "I like this apple."
        ],
        "y": [
            0, # 0 for negative
            1, # 1 for positive
        ]
    })

    print("Start attack")
    attack_eval = OpenAttack.AttackEval(attacker, clsf, metrics = [
        OpenAttack.metric.EditDistance(),
        OpenAttack.metric.ModificationRate()
    ])
    attack_eval.eval(dataset, visualize=True)

if __name__ == "__main__":
    main()
