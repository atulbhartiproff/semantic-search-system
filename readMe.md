# Semantic Search Engine

Gonna build something like this :

Dataset (20k documents)
        |
        V
Text preprocessing
        |
        V
Embedding model -> vector representations
        |
        V
Vector database (store embeddings)
        |
        V
Fuzzy clustering (documents belong to multiple clusters)
        |
        V
Semantic cache (for queries)
        |
        V
FastAPI service
        |
        V
User queries -> cached semantic search results