package com.vectordb.algorithms;

import com.vectordb.model.SearchResult;
import com.vectordb.model.VectorItem;
import com.vectordb.util.DistanceMetric;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;

/**
 * Brute Force k-NN Search — O(N·d) time, O(1) extra space.
 *
 * Role in this project:
 * 1. Correctness baseline — used to verify HNSW recall@k
 * 2. Ground truth for benchmarking comparisons
 * 3. Fallback when the index is very small (< 10 items)
 *
 * Thread safety: CopyOnWriteArrayList gives safe concurrent reads.
 * Writes are rare (inserts/deletes), so COW overhead is acceptable.
 */
public class BruteForce {

    // CopyOnWriteArrayList: reads need no lock; writes copy the whole array.
    // Ideal when reads >> writes, which is the dominant pattern in a vector DB.
    private final CopyOnWriteArrayList<VectorItem> items = new CopyOnWriteArrayList<>();

    public void insert(VectorItem item) {
        items.add(item);
    }

    /**
     * Exact k-nearest-neighbour search.
     * Iterates all items, computes distance, keeps top-k via sort.
     *
     * @param query  query vector (must match item dimensions)
     * @param k      number of neighbours to return
     * @param metric distance function
     * @return sorted list of (distance, id) pairs, ascending
     */
    public List<SearchResult> knn(float[] query, int k, DistanceMetric metric) {
        List<SearchResult> results = new ArrayList<>(items.size());

        // Snapshot iteration — CopyOnWriteArrayList guarantees consistent snapshot
        for (VectorItem item : items) {
            float dist = metric.compute(query, item.getEmbedding());
            results.add(new SearchResult(item.getId(), dist));
        }

        Collections.sort(results);          // O(N log N)
        return results.subList(0, Math.min(k, results.size()));
    }

    public void remove(int id) {
        items.removeIf(item -> item.getId() == id);
    }

    public int size() {
        return items.size();
    }

    public List<VectorItem> getAll() {
        return new ArrayList<>(items);
    }
}
