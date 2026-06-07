# VectorDB — Vector Database Engine in Java

> Implements **HNSW**, **KD-Tree**, and **Brute Force** vector search from scratch in Java.
> Built to understand how production databases like Pinecone, Weaviate, and Chroma work under the hood.

[![Java](https://img.shields.io/badge/Java-17-orange)](https://openjdk.org/projects/jdk/17/)
[![Spring Boot](https://img.shields.io/badge/Spring_Boot-3.2-green)](https://spring.io/projects/spring-boot)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

---

## Benchmark Results

> Measured on local machine · 16D vectors · k=10 · cosine distance · 20 runs avg after warmup

| N (vectors) | BruteForce (μs) | KD-Tree (μs) | HNSW (μs) | HNSW Speedup |
|------------|----------------|-------------|-----------|-------------|
| 100        | ~12            | ~8          | ~6        | ~2x         |
| 1,000      | ~95            | ~18         | ~9        | ~10x        |
| 5,000      | ~480           | ~32         | ~14       | ~34x        |
| 10,000     | ~960           | ~51         | ~18       | ~53x        |

> Run `mvn test -Dtest=BenchmarkTest` to generate exact numbers on your machine.

**Recall@10** (HNSW vs brute force ground truth):

| N      | Recall@10 |
|--------|-----------|
| 200    | ~0.97     |
| 500    | ~0.95     |
| 1,000  | ~0.93     |
| 2,000  | ~0.91     |

---

## Architecture

```
vectordb-java/
├── algorithms/
│   ├── HNSW.java          ← Multilayer graph, O(log N) search
│   ├── KDTree.java        ← Binary space partition, exact, O(log N) at low-D
│   └── BruteForce.java    ← Exact baseline, used for recall testing
│
├── storage/
│   └── WAL.java           ← Write-Ahead Log (crash-safe persistence)
│
├── core/
│   ├── VectorDB.java      ← Unified index over all 3 algorithms + WAL
│   ├── DocumentDB.java    ← HNSW index for Ollama embeddings (768D)
│   └── OllamaClient.java  ← HTTP client → embed + generate
│
└── api/
    ├── SearchController.java   ← Search, benchmark, recall endpoints
    ├── DocumentController.java ← RAG pipeline endpoints
    └── StatusController.java   ← Health + status
```

---

## How HNSW Works

```
Insert:
  node → random layer assignment (exponential distribution)
  → greedy descent from topLayer to node's layer (ef=1)
  → beam search at each layer (ef=efConstruct=200)
  → connect to M nearest neighbors (bidirectional)

Search:
  → greedy descent from topLayer to layer 1 (ef=1, fast highway)
  → beam search at layer 0 (ef=50, precise zoom-in)
  → return top-k
```

Upper layers act as a "highway" — fast long-range navigation.
Layer 0 has all nodes with dense connections for precise local search.
This is why HNSW achieves O(log N) instead of O(N) for brute force.

---

## Why KD-Tree Loses at High Dimensions

KD-Tree pruning relies on axis-aligned distance bounds. In high dimensions (d >> 20),
nearly all space lies near the boundary of the hypersphere — almost no subtrees get pruned.
At 768D (Ollama embeddings), KD-Tree degrades to near-brute-force.
HNSW's graph-based approach doesn't suffer from this — it's why HNSW dominates in production.

---

## RAG Pipeline

```
User question
    → nomic-embed-text (768D embedding via Ollama)
    → HNSW search (DocumentDB)
    → top-3 chunks retrieved
    → prompt assembled: [context] + [question]
    → llama3.2 generates answer
    → streamed back to UI
```

---

## Key Engineering Decisions (Interview Topics)

| Decision | What I chose | Why |
|---|---|---|
| Thread safety | `ReentrantReadWriteLock` on HNSW | Many concurrent reads, rare writes |
| ID generation | `AtomicInteger` | Lock-free, no contention on insert hot path |
| Persistence | Write-Ahead Log (WAL) | Same model as PostgreSQL — durable before ACK |
| BruteForce storage | `CopyOnWriteArrayList` | Reads are dominant, COW is correct |
| HTTP server | Spring Boot | Industry standard, auto-configuration |
| Embedding client | Java `HttpClient` (JDK 11+) | No extra dependencies needed |

---

## Prerequisites

- Java 17+
- Maven 3.8+
- [Ollama](https://ollama.com) (for RAG features — optional)

```bash
# Pull Ollama models (optional, for RAG tab)
ollama pull nomic-embed-text   # 274MB — embedding model
ollama pull llama3.2           # 2GB   — language model
```

---

## Quick Start

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/vectordb-java.git
cd vectordb-java

# Run
./mvnw spring-boot:run

# Open browser
open http://localhost:8080
```

### One-command with Docker

```bash
docker-compose up
# Open http://localhost:8080
```

---

## REST API

### Demo Vector Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/search?v=f1,f2,...&k=5&metric=cosine&algo=hnsw` | k-NN search |
| `POST` | `/insert` | Insert a demo vector |
| `DELETE` | `/delete/{id}` | Delete by ID |
| `GET`  | `/items` | List all vectors |
| `GET`  | `/benchmark?v=...&k=5&metric=cosine` | Compare all 3 algorithms |
| `GET`  | `/hnsw-info` | HNSW graph structure |
| `GET`  | `/recall?k=10` | Compute recall@k vs brute force |
| `GET`  | `/stats` | Database statistics |

### Document & RAG Endpoints

| Method | Endpoint | Body | Description |
|--------|----------|------|-------------|
| `POST` | `/doc/insert` | `{"title":"...","text":"..."}` | Embed & store document |
| `GET`  | `/doc/list` | — | List all documents |
| `DELETE` | `/doc/delete/{id}` | — | Delete document |
| `POST` | `/doc/ask` | `{"question":"...","k":3}` | Full RAG pipeline |
| `GET`  | `/status` | — | Ollama + DB status |

### Example

```bash
# Search
curl "http://localhost:8080/search?v=0.9,0.8,0.7,0.6,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1&k=3&metric=cosine&algo=hnsw"

# Ask a question (requires Ollama + documents inserted)
curl -X POST http://localhost:8080/doc/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What is dynamic programming?","k":3}'

# Get recall@10 stats
curl "http://localhost:8080/recall?k=10"
```

---

## Running Tests

```bash
# All tests
mvn test

# Just benchmarks (prints speed tables)
mvn test -Dtest=BenchmarkTest

# Just HNSW correctness + recall
mvn test -Dtest=HNSWTest

# Just WAL crash recovery
mvn test -Dtest=WALTest
```

---

## What I Learned Building This

- **HNSW internals** — random level assignment, beam search, bidirectional linking, why ef_construction vs ef_search is a recall/latency tradeoff
- **Curse of dimensionality** — why KD-Tree becomes brute force above 20 dimensions
- **Write-Ahead Logs** — the fundamental durability primitive in every database (PostgreSQL, MySQL, RocksDB all use this)
- **Java concurrency** — `ReentrantReadWriteLock` for multiple readers / exclusive writers, `AtomicInteger` for lock-free ID generation, `CopyOnWriteArrayList` for read-heavy structures
- **Recall@k as a metric** — the standard from ann-benchmarks.com used to evaluate vector indexes in production

---

## References

- [HNSW Paper — Malkov & Yashunin (2018)](https://arxiv.org/abs/1603.09320)
- [ann-benchmarks.com — Standard recall benchmarking](https://ann-benchmarks.com)
- [Designing Data-Intensive Applications — Martin Kleppmann](https://dataintensive.net) (WAL chapter)
