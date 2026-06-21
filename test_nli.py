from transformers import pipeline

# Load an NLI model
# This will download ~500MB the first time; be patient
nli = pipeline("text-classification", model="cross-encoder/nli-deberta-v3-base")

def check_contradiction(premise: str, hypothesis: str):
    # NLI models expect input as "premise [SEP] hypothesis"
    result = nli(f"{premise} [SEP] {hypothesis}")
    return result[0]  # returns {'label': '...', 'score': ...}

# Test pairs
pairs = [
    ("Alice lives in Paris.", "Alice resides in France."),         # entailment
    ("Alice lives in Paris.", "Alice lives in Tokyo."),             # contradiction
    ("Alice lives in Paris.", "Alice likes coffee."),               # neutral
    ("Alice lives at 123 Main Street.", "Alice lives at 456 Oak Ave."), # contradiction
]

for premise, hypothesis in pairs:
    result = check_contradiction(premise, hypothesis)
    print(f"P: {premise}")
    print(f"H: {hypothesis}")
    print(f"   → {result['label']} (score: {result['score']:.3f})\n")