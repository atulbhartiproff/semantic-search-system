# Semantic Search Engine with Fuzzy Clustering and Semantic Cache

## Overview

This project implements a lightweight **semantic search system** built on the **20 Newsgroups dataset**. The system uses vector embeddings, fuzzy clustering, and a custom semantic cache to provide efficient search capabilities through a FastAPI service.

The primary goal is to demonstrate how modern semantic search pipelines can be designed while focusing on **system design decisions**, **efficient retrieval**, and **clean API architecture**.

---

## Key Features

* Semantic search using transformer-based embeddings
* Vector similarity search with FAISS
* Fuzzy clustering of documents using Gaussian Mixture Models
* Custom semantic cache for avoiding redundant query computation
* Cluster-aware cache lookup for efficiency
* FastAPI service exposing a clean API
* Dockerized environment for reproducibility

---

## System Architecture

```
Dataset (20 Newsgroups)
        ↓
Text preprocessing
        ↓
Embedding model (Sentence Transformers)
        ↓
Vector database (FAISS)
        ↓
Fuzzy clustering (Gaussian Mixture Model)
        ↓
Semantic cache layer
        ↓
FastAPI service
```

---

## Technologies Used

* Python
* FastAPI
* Sentence Transformers
* FAISS
* Scikit-learn
* NumPy
* Docker

---

## Dataset

The system uses the **20 Newsgroups dataset**, which contains approximately 20,000 newsgroup posts across 20 different categories.

The dataset contains overlapping topics such as:

* Space
* Politics
* Religion
* Computers
* Firearms

Because the topics overlap, fuzzy clustering is used instead of hard clustering.

---

## Fuzzy Clustering

Instead of assigning each document to a single cluster, the system uses **Gaussian Mixture Models** to assign **probability distributions across clusters**.

Example:

```
Document: "Gun policy debate"

Cluster probabilities:
Politics → 0.58
Firearms → 0.37
Other → 0.05
```

This better represents real-world semantic overlap between topics.

---

## Semantic Cache

Traditional caching fails when queries are phrased differently.

Example:

```
"space shuttle launch"
"nasa rocket launch"
```

Both queries may have the same semantic meaning.

This project implements a **semantic cache** that:

1. Embeds incoming queries
2. Compares them to previously seen queries using cosine similarity
3. Returns cached results if similarity exceeds a threshold

### Cache Strategy

* Cosine similarity for query comparison
* Configurable similarity threshold
* Cluster-aware cache lookup for faster search

---

## API Endpoints

### POST /query

Accepts a natural language query and returns a semantic search result.

Example request:

```
{
  "query": "space shuttle launch"
}
```

Example response:

```
{
  "query": "space shuttle launch",
  "cache_hit": false,
  "matched_query": null,
  "similarity_score": null,
  "result": "...document text...",
  "dominant_cluster": 3
}
```

---

### GET /cache/stats

Returns statistics about the cache.

Example response:

```
{
  "total_entries": 10,
  "hit_count": 4,
  "miss_count": 6,
  "hit_rate": 0.4
}
```

---

### DELETE /cache

Clears the semantic cache and resets statistics.

---

## Project Structure

```
semantic-search-engine/

api/
    main.py

cache/
    semantic_cache.py

clustering/
    fuzzy_cluster.py

data/
    dataset_loader.py

embedding/
    embedding_model.py

retrieval/
    search_engine.py

vectorstore/
    faiss_index.py

scripts/
    precompute.py

artifacts/
    (generated embeddings, clustering models, and index)

Dockerfile

docker-compose.yml

requirements.txt
```

---

## Running the Project

### Local Setup

Create a virtual environment:

```
python -m venv venv
```

Activate the environment:

```
source venv/bin/activate
```

Install dependencies:

```
pip install -r requirements.txt
```

Run the API:

```
uvicorn api.main:app --reload
```

API will be available at:

```
http://localhost:8000/docs
```

---

## Docker Setup

Build and run the container:

```
docker-compose up --build
```

The Docker build step performs precomputation of embeddings, clustering models, and FAISS indices to reduce API startup latency.

After startup, the API will be available at:

```
http://localhost:8000/docs
```

---

## Performance Optimization

To reduce startup time, embeddings and clustering artifacts are precomputed during the Docker build phase and saved as serialized files.

This allows the API to load the system state quickly rather than recomputing embeddings each time the container starts.

---

## Design Decisions

### Embedding Model

`all-MiniLM-L6-v2` was selected because it provides strong semantic performance while remaining lightweight and fast.

### Vector Database

FAISS was chosen for efficient local vector similarity search.

### Clustering Algorithm

Gaussian Mixture Models allow probabilistic cluster membership, enabling fuzzy clustering.

### Cache Strategy

Cosine similarity with a configurable threshold determines whether a query should reuse a cached result.

Cluster-aware cache lookup reduces search overhead when the cache grows.

---

## Future Improvements

Potential improvements include:

* Hybrid retrieval combining vector search with keyword scoring
* Cache eviction policies for large-scale deployments
* Cluster visualization tools
* Query result ranking improvements

---

## License

This project
