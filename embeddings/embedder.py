from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# STEP 1: Load the embedding model
# --------------------------------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded successfully!")


# --------------------------------------------------
# STEP 2: Create some example sentences
# --------------------------------------------------

sentences = [
    "Cybersecurity is important in software development.",
    "Security should be integrated into the software development lifecycle.",
    "The weather is sunny today."
]


# --------------------------------------------------
# STEP 3: Convert sentences into vectors
# --------------------------------------------------

embeddings = model.encode(sentences)


# --------------------------------------------------
# STEP 4: Display the results
# --------------------------------------------------

print("\nNumber of sentences:", len(sentences))

print("Embedding shape:", embeddings.shape)

print("\nFirst sentence:")
print(sentences[0])

print("\nFirst sentence vector:")
print(embeddings[0])