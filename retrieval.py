from backend.main import extract_pages_from_pdf, chunk_text
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

PDF_PATH = "backend/uploads/EduLex_SE_Lab_Report.pdf"

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Extract pages
pages = extract_pages_from_pdf(PDF_PATH)

# Create document ID for this test
document_id = "test-document"

# Create chunks
chunks = chunk_text(
    pages,
    document_id,
    "EduLex_SE_Lab_Report.pdf"
)

print("Number of chunks:", len(chunks))

# Extract only the text from each chunk
chunk_texts = [chunk["text"] for chunk in chunks]

# Create embeddings for all chunks
chunk_embeddings = model.encode(chunk_texts)

print("Number of embeddings:", len(chunk_embeddings))
print("Embedding dimensions:", len(chunk_embeddings[0]))
# User's question
query = "What technologies are used in the EduLex system?"

# Convert the question into an embedding
query_embedding = model.encode([query])

# Calculate similarity between query and every chunk
similarities = cosine_similarity(
    query_embedding,
    chunk_embeddings
)[0]

# Get indices of the top 3 most similar chunks
top_indices = similarities.argsort()[-10:][::-1]

print("\nQuery:")
print(query)

print("\nTop 10 relevant chunks:\n")

for rank, index in enumerate(top_indices, start=1):
    print(f"Rank {rank}")
    print(f"Similarity: {similarities[index]:.4f}")
    print(f"Chunk index: {chunks[index]['chunk_index']}")
    print(
    f"Pages: {chunks[index]['start_page']}"
    f"-{chunks[index]['end_page']}"
    )
    print(f"Text: {chunks[index]['text'][:500]}")
    print("-" * 80)