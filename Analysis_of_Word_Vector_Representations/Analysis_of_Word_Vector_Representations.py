import json
import numpy as np
from numpy.linalg import norm

# =========================
# STEP 1: LOAD DATA
# =========================

with open(r"C:\Users\Charl\Desktop\tiny_glove.json", "r") as f:
    glove = json.load(f)

print("Vocabulary size:", len(glove))

sample_words = list(glove.keys())[:20]
print("\nFirst 20 words:")
print(sample_words)


# =========================
# STEP 2: INSPECT VECTOR
# =========================

word = "king"
vector = np.array(glove[word])

print("\nWord:", word)
print("Vector shape:", vector.shape)
print("First 10 values:", vector[:10])


# =========================
# STEP 3: UTILITIES
# =========================

def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

def get_vector(word):
    if word in glove:
        return np.array(glove[word])
    return None


# =========================
# STEP 4: SIMILARITY PAIRS
# =========================

pairs = [
    ("king", "queen"),
    ("man", "woman"),
    ("doctor", "nurse"),
    ("king", "apple"),
    ("teacher", "rich")
]

print("\nCosine similarities:")

for w1, w2 in pairs:
    v1 = get_vector(w1)
    v2 = get_vector(w2)

    if v1 is not None and v2 is not None:
        sim = cosine_similarity(v1, v2)
        print(f"{w1:10s} vs {w2:10s} -> {sim:.4f}")
    else:
        print(f"Missing word: {w1} or {w2}")


# =========================
# STEP 5: NEAREST WORDS
# =========================

def nearest_words(target_word, top_n=10):
    if target_word not in glove:
        return []

    target_vec = get_vector(target_word)

    scores = []

    for word in glove:
        if word == target_word:
            continue

        sim = cosine_similarity(target_vec, get_vector(word))
        scores.append((word, sim))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_n]


target = "king"

print("\nNearest words to:", target)
for word, score in nearest_words(target):
    print(f"{word:15s} {score:.4f}")


# =========================
# STEP 6: PROFESSIONS
# =========================

profession_words = ["doctor", "nurse", "engineer", "teacher"]

for base_word in profession_words:
    print(f"\nNearest to {base_word}:")
    for word, score in nearest_words(base_word, top_n=5):
        print(f"{word:15s} {score:.4f}")


# =========================
# STEP 7: WORD ARITHMETIC
# =========================

result_vector = (
    get_vector("king")
    - get_vector("man")
    + get_vector("woman")
)

scores = []

for word in glove:
    sim = cosine_similarity(result_vector, get_vector(word))
    scores.append((word, sim))

scores.sort(key=lambda x: x[1], reverse=True)

print("\nking - man + woman ≈ ?\n")

for word, score in scores[:10]:
    print(f"{word:15s} {score:.4f}")


# =========================
# STEP 8: MORE ARITHMETIC
# =========================

experiments = [
    ("queen", "woman", "man"),
    ("doctor", "man", "woman"),
    ("teacher", "man", "woman")
]

for a, b, c in experiments:
    print(f"\nTesting: {a} - {b} + {c}")

    vec = get_vector(a) - get_vector(b) + get_vector(c)

    scores = []

    for word in glove:
        sim = cosine_similarity(vec, get_vector(word))
        scores.append((word, sim))

    scores.sort(key=lambda x: x[1], reverse=True)

    for word, score in scores[:5]:
        print(f"{word:15s} {score:.4f}")


# =========================
# STEP 9: SENTENCE VECTOR
# =========================

def sentence_vector(sentence):
    words = sentence.lower().split()
    vectors = []

    for word in words:
        if word in glove:
            vectors.append(get_vector(word))

    if len(vectors) == 0:
        return np.zeros(50)

    return np.mean(vectors, axis=0)


sentence = "king queen man woman"
vec = sentence_vector(sentence)

print("\nSentence:", sentence)
print("Vector shape:", vec.shape)
print("First 10 values:", vec[:10])


# =========================
# STEP 10: SENTENCE SIMILARITY
# =========================

sentences = [
    "king queen",
    "man woman",
    "doctor nurse",
    "banana orange"
]

base_sentence = "king man"
base_vec = sentence_vector(base_sentence)

print("\nBase sentence:", base_sentence)

for s in sentences:
    vec = sentence_vector(s)
    sim = cosine_similarity(base_vec, vec)
    print(f"{s:15s} -> {sim:.4f}")


# =========================
# STEP 11: OOV CHECK
# =========================

test_words = ["king", "dragon", "teacher", "spaceship"]

print("\nOut-of-vocabulary check:")

for word in test_words:
    if word in glove:
        print(f"{word:10s} -> Found")
    else:
        print(f"{word:10s} -> Missing")


# =========================
# STEP 12: SIMILARITY TABLE
# =========================

words = ["king", "queen", "man", "woman"]

print("\nCosine Similarity Table:\n")

for w1 in words:
    row = []
    for w2 in words:
        sim = cosine_similarity(get_vector(w1), get_vector(w2))
        row.append(f"{sim:.3f}")
    print(w1.ljust(8), row)


# =========================
# STEP 13: FINAL EXPLORATION
# =========================

custom_words = ["science", "technology", "teacher", "student"]

for word in custom_words:
    if word in glove:
        print(f"\nTop neighbors for {word}:")
        for neighbor, score in nearest_words(word, top_n=5):
            print(f"{neighbor:15s} {score:.4f}")
    else:
        print(f"\n{word} not found in vocabulary.")
