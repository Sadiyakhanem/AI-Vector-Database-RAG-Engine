package com.vectordb.core;

import com.vectordb.algorithms.BruteForce;
import com.vectordb.algorithms.HNSW;
import com.vectordb.model.SearchResult;
import com.vectordb.model.VectorItem;
import com.vectordb.util.DistanceMetric;
import org.springframework.stereotype.Component;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * DocumentDB — HNSW index over real Ollama embeddings (768D).
 *
 * Separate from VectorDB because:
 *   1. Different dimensionality (768D vs 16D demo)
 *   2. Different use case — documents are chunked text, not structured vectors
 *   3. Allows independent scaling decisions
 *
 * Chunking strategy:
 *   250 words per chunk with 30-word overlap.
 *   Overlap prevents context loss at chunk boundaries — a technique used
 *   by LangChain, LlamaIndex, and every production RAG system.
 */
@Component
public class DocumentDB {

    public record DocItem(int id, String title, String text, float[] embedding) {}

    private final ConcurrentHashMap<Integer, DocItem> store = new ConcurrentHashMap<>();
    private final HNSW       hnsw       = new HNSW(16, 200);
    private final BruteForce bruteForce = new BruteForce();
    private final AtomicInteger nextId  = new AtomicInteger(1);
    private volatile int dims = 0;

    public int insert(String title, String text, float[] embedding) {
        if (dims == 0) dims = embedding.length;

        int id = nextId.getAndIncrement();
        DocItem item = new DocItem(id, title, text, embedding);
        store.put(id, item);

        VectorItem vi = new VectorItem(id, title, "doc", embedding);
        hnsw.insert(vi, DistanceMetric.cosine());
        bruteForce.insert(vi);

        return id;
    }

    public List<DocSearchResult> search(float[] query, int k, float maxDist) {
        if (store.isEmpty()) return Collections.emptyList();

        List<SearchResult> raw = store.size() < 10
            ? bruteForce.knn(query, k, DistanceMetric.cosine())
            : hnsw.knn(query, k, 50, DistanceMetric.cosine());

        List<DocSearchResult> results = new ArrayList<>();
        for (SearchResult r : raw) {
            if (r.getDistance() <= maxDist) {
                DocItem doc = store.get(r.getId());
                if (doc != null) results.add(new DocSearchResult(r.getDistance(), doc));
            }
        }
        return results;
    }

    public boolean delete(int id) {
        if (!store.containsKey(id)) return false;
        store.remove(id);
        hnsw.remove(id);
        bruteForce.remove(id);
        return true;
    }

    public Collection<DocItem> getAll()  { return Collections.unmodifiableCollection(store.values()); }
    public int                 size()    { return store.size(); }
    public int                 getDims() { return dims; }

    /**
     * Chunk text into overlapping windows.
     * Static utility — pure function, easy to unit test.
     *
     * @param text         input text
     * @param chunkWords   words per chunk
     * @param overlapWords words shared between adjacent chunks
     */
    public static List<String> chunkText(String text, int chunkWords, int overlapWords) {
        String[] words = text.trim().split("\\s+");
        if (words.length == 0) return Collections.emptyList();
        if (words.length <= chunkWords) return List.of(text.trim());

        List<String> chunks = new ArrayList<>();
        int step = chunkWords - overlapWords;
        for (int i = 0; i < words.length; i += step) {
            int end = Math.min(i + chunkWords, words.length);
            chunks.add(String.join(" ", Arrays.copyOfRange(words, i, end)));
            if (end == words.length) break;
        }
        return chunks;
    }

    public record DocSearchResult(float distance, DocItem doc) {}
}
