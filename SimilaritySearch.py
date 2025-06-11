import numpy as np
from sentence_transformers import SentenceTransformer

# Initialize the sentence transformer model for semantic embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Example Q&A dictionary (can be from any domain)
qa_dict = {
    "What is the basic unit of life?": "The cell is the basic unit of life.",
    "What are the two main types of cells?": "Prokaryotic and eukaryotic cells.",
    "What structures do plant cells have that animal cells do not?": "Plant cells have a cell wall and chloroplasts, which animal cells do not have.",
    "What is the function of the nucleus?": "The nucleus controls the cell's activities and contains DNA.",
    "What is photosynthesis?": "Photosynthesis is the process by which plants convert light energy into chemical energy.",
    "What is the function of mitochondria?": "Mitochondria produce energy for the cell in the form of ATP.",
    "What is the primary difference between plant and animal cells?": "Plant cells have chloroplasts and a cell wall; animal cells do not.",
}

# Extract questions and answers explicitly
questions = []
answers = []
for q in qa_dict:
    questions.append(q)
    answers.append(qa_dict[q])

# Encode questions into embeddings and normalize for cosine similarity
question_embeddings = model.encode(questions)
def normalize_vectors(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms
question_embeddings = normalize_vectors(question_embeddings)

def semantic_search(query, top_k=3):
    """
    Compute cosine similarity between the query and all stored questions,
    and return top_k most similar questions with their answers and scores.
    """
    query_vec = model.encode([query])[0]
    query_vec = query_vec / np.linalg.norm(query_vec)

    similarities = []
    for i in range(len(question_embeddings)):
        sim = np.dot(query_vec, question_embeddings[i])
        similarities.append(sim)

    # Select top_k results by similarity score
    sorted_indices = np.argsort(similarities)[::-1]
    top_indices = []
    count = 0
    for idx in sorted_indices:
        top_indices.append(idx)
        count += 1
        if count == top_k:
            break

    results = []
    for idx in top_indices:
        res = {
            "question": questions[idx],
            "answer": answers[idx],
            "score": float(similarities[idx])
        }
        results.append(res)
    return results

if __name__ == "__main__":
    print(
        "Welcome to the Semantic Search QA system!\n"
        "This program uses a pretrained sentence transformer model\n"
        "to convert questions into semantic vectors and compare their similarity\n"
        "using cosine similarity. When you ask a question, it finds the most semantically\n"
        "similar stored questions and returns their answers.\n"
        "Type 'exit' or 'quit' to end the program."
    )

    while True:
        user_query = input("\nYour question: ").strip()
        if user_query.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        if user_query == "":
            print("Please enter a non-empty question.")
            continue

        search_results = semantic_search(user_query, top_k=2)
        print(f"\nTop {len(search_results)} matching results:")
        for i in range(len(search_results)):
            print(f"{i+1}. Question: {search_results[i]['question']}")
            print(f"   Answer: {search_results[i]['answer']}")
            print(f"   Similarity score: {search_results[i]['score']:.4f}\n")
