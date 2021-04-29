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
    clsf = OpenAttack.classifiers.HuggingfaceClassifier(model, tokenizer=tokenizer, max_len=100, embedding_layer=model.bert.embeddings.word_embeddings)

    print("New Attacker")
    attacker = OpenAttack.attackers.PWWSAttacker()

    dataset = datasets.load_dataset("sst", split="train[:20]").map(function=dataset_mapping)

    print("Start attack")
    options = {
        "success_rate": True,
        "fluency": False,
        "mistake": False,
        "semantic": False,
        "levenstein": True,
        "word_distance": False,
        "modification_rate": True,
        "running_time": True,

        "invoke_limit": 500,
        "average_invoke": True
    }
    attack_eval = OpenAttack.attack_evals.InvokeLimitedAttackEval(attacker, clsf, **options )
    attack_eval.eval(dataset, visualize=True)

if __name__ == "__main__":
    main()
