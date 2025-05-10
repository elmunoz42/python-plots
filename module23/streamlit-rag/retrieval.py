# retrieval.py
def get_similar_documents(collection_retrieved, prompt_vector):
    response = collection_retrieved.query(
            query_embeddings=[prompt_vector],
            n_results=20,
            include=['documents', 'metadatas'], #'distances','embeddings',
            )
    return response