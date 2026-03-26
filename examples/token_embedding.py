import os
from dotenv import load_dotenv
from google import genai
import tiktoken
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# 1. Load API Key
# -----------------------------
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# -----------------------------
# 2. TOKENIZATION
# -----------------------------
print("\n===== TOKENIZATION =====")

enc = tiktoken.encoding_for_model("gpt-4o-mini")

sample_text = "Agentic AI agents can plan, reason, and use tools."

tokens = enc.encode(sample_text)

print("Original text:", sample_text)
print("Number of tokens:", len(tokens))
print("Token IDs:", tokens)
print("Decoded tokens:", [enc.decode([t]) for t in tokens])

# -----------------------------
# 3. TOKEN COMPARISON
# -----------------------------
print("\n===== TOKEN COMPARISON =====")

sentences = [
    "AI is amazing.",
    "Artificial Intelligence is amazing.",
    "AI is 🔥."
]

for s in sentences:
    t = enc.encode(s)
    print(f"\nText: {s}")
    print(f"Tokens: {len(t)} → {t}")


# -----------------------------
# 4. EMBEDDINGS (LOCAL ✅)
# -----------------------------
print("\n===== EMBEDDINGS =====")

from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

texts = ["Im fine", "I am happy", "I am sad","how are you", "Bananas are yellow"]   

embeddings = model.encode(texts)

for i, vec in enumerate(embeddings):
    print(f"{texts[i]} → vector length: {len(vec)}")

# -----------------------------
# 5. PCA VISUALIZATION
# -----------------------------
print("\n===== PCA VISUALIZATION =====")

pca = PCA(n_components=2)
points = pca.fit_transform(embeddings)

plt.figure(figsize=(6,6))

for i, txt in enumerate(texts):
    plt.scatter(points[i, 0], points[i, 1])
    plt.text(points[i, 0], points[i, 1], txt)

plt.title("Embedding Visualization (2D)")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid()

plt.show()

# -----------------------------
# 6. SIMILARITY
# -----------------------------
print("\n===== COSINE SIMILARITY =====")

sim = cosine_similarity([embeddings[0]], embeddings)

print("Comparing:", texts[0])
for i, score in enumerate(sim[0]):
    print(f"Similarity with '{texts[i]}': {score:.4f}")
