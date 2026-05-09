import pandas as pd
from transformers import pipeline

# =========================
# Load dataset
# =========================
df = pd.read_csv("imdb_top_500.csv")

reviews = df["text"].iloc[:50].tolist()
true_labels = df["label"].iloc[:50].tolist()

# =========================
# Models to test
# =========================
models = {
    "DistilBERT SST2": "distilbert-base-uncased-finetuned-sst-2-english",
    "RoBERTa Twitter": "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "BERT SST2": "textattack/bert-base-uncased-SST-2"
}

# =========================
# Convert predictions to 0/1
# =========================
def to_binary(pred, model_name):
    label = pred["label"].upper()

    # DistilBERT / SST2
    if "POSITIVE" in label:
        return 1
    if "NEGATIVE" in label:
        return 0

    # RoBERTa Twitter (LABEL_0, LABEL_1, LABEL_2)
    if "LABEL_0" in label:
        return 0
    if "LABEL_1" in label:
        return 1
    if "LABEL_2" in label:
        return 1

    return 0

# =========================
# Evaluation loop
# =========================
results = {}

for name, model_name in models.items():
    print("\n==============================")
    print("Testing model:", name)
    print("==============================")

    try:
        classifier = pipeline(
            "sentiment-analysis",
            model=model_name,
            tokenizer=model_name,
            framework="pt",
            truncation=True,
            max_length=512
        )

        predictions = classifier(reviews)

        pred_labels = [to_binary(p, name) for p in predictions]

        correct = sum(p == y for p, y in zip(pred_labels, true_labels))
        accuracy = correct / len(true_labels)

        results[name] = accuracy

        print("Accuracy:", round(accuracy, 4))

    except Exception as e:
        print("ERROR:", str(e))
        results[name] = None

# =========================
# Final summary
# =========================
print("\n==============================")
print("FINAL RESULTS")
print("==============================")

for k, v in results.items():
    print(f"{k}: {v}")
