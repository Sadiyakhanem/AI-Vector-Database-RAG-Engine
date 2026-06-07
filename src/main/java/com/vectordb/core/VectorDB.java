package com.vectordb.core;

import com.vectordb.algorithms.BruteForce;
import com.vectordb.algorithms.HNSW;
import com.vectordb.algorithms.KDTree;
import com.vectordb.model.SearchResult;
import com.vectordb.model.VectorItem;
import com.vectordb.storage.WAL;
import com.vectordb.util.DistanceMetric;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import java.util.logging.Logger;

/**
 * VectorDB — Unified interface over HNSW, KD-Tree, and BruteForce.
 *
 * Responsibilities:
 *   1. ID management (atomic counter — thread-safe, no locks needed)
 *   2. WAL writes (before every mutation — crash safety)
 *   3. Dispatching to the correct algorithm
 *   4. Maintaining the authoritative item store
 *   5. WAL replay on startup
 *
 * Why three separate indexes?
 *   They serve different purposes:
 *   - BruteForce: ground truth for recall testing, fallback for tiny datasets
 *   - KDTree: exact search, fast for 16D demo vectors
 *   - HNSW: approximate search, scales to millions of vectors
 *
 * Thread safety:
 *   The item store uses a ConcurrentHashMap.
 *   Each algorithm manages its own internal locking.
 *   The WAL is synchronized internally.
 *   AtomicInteger for ID generation avoids any lock on the hot path.
 */
@Component
public class VectorDB {

    private static final Logger log = Logger.getLogger(VectorDB.class.getName());

    // ── Storage ──────────────────────────────────────────────────────────
    private final ConcurrentHashMap<Integer, VectorItem> store = new ConcurrentHashMap<>();

    // ── Algorithms ───────────────────────────────────────────────────────
    private final BruteForce bruteForce;
    private final KDTree     kdTree;
    private final HNSW       hnsw;

    // ── ID counter ───────────────────────────────────────────────────────
    // AtomicInteger: lock-free increment. No contention on the insert hot path.
    private final AtomicInteger nextId = new AtomicInteger(1);

    // ── WAL ──────────────────────────────────────────────────────────────
    private WAL wal;

    // ── KDTree rebuild lock ───────────────────────────────────────────────
    // KDTree doesn't support single-item delete; we rebuild on delete.
    // This lock prevents concurrent rebuilds.
    private final ReadWriteLock kdtLock = new ReentrantReadWriteLock();

    private final int dims;

    public VectorDB(int dims) {
        this.dims       = dims;
        this.bruteForce = new BruteForce();
        this.kdTree     = new KDTree(dims);
        this.hnsw       = new HNSW(16, 200);

        initWAL();
        replayWAL();
    }

    // ── WAL Setup ────────────────────────────────────────────────────────

    private void initWAL() {
        try {
            this.wal = new WAL("vectordb.wal");
        } catch (IOException e) {
            log.warning("Could not initialize WAL — running without persistence: " + e.getMessage());
        }
    }

    /**
     * Replay WAL on startup to restore state after a crash or restart.
     * This is the same recovery mechanism PostgreSQL uses.
     */
    private void replayWAL() {
        if (wal == null) return;
        try {
            List<WAL.WALEntry> entries = wal.replay();
            int maxId = 0;
            for (WAL.WALEntry e : entries) {
                if (e.op() == WAL.WALEntry.Op.INSERT && e.embedding() != null) {
                    VectorItem item = new VectorItem(e.id(), e.metadata(), e.category(), e.embedding());
                    store.put(e.id(), item);
                    bruteForce.insert(item);
                    hnsw.insert(item, DistanceMetric.cosine());
                    maxId = Math.max(maxId, e.id());
                } else if (e.op() == WAL.WALEntry.Op.DELETE) {
                    store.remove(e.id());
                    bruteForce.remove(e.id());
                    hnsw.remove(e.id());
                    // KDTree will be rebuilt below
                }
            }

            if (!entries.isEmpty()) {
                // Rebuild KDTree once after replay (cheaper than per-entry rebuild)
                kdTree.rebuild(new ArrayList<>(store.values()));
                nextId.set(maxId + 1);
                log.info("WAL replay complete — restored " + store.size() + " items, nextId=" + nextId.get());
            }
        } catch (IOException e) {
            log.warning("WAL replay failed: " + e.getMessage());
        }
    }

    // ── Insert ───────────────────────────────────────────────────────────

    public VectorItem insert(String metadata, String category,
                             float[] embedding, DistanceMetric metric) {
        int id = nextId.getAndIncrement();
        VectorItem item = new VectorItem(id, metadata, category, embedding);

        // WAL FIRST — write to disk before updating memory
        if (wal != null) {
            try {
                wal.logInsert(id, category, metadata, embedding);
            } catch (IOException e) {
                log.warning("WAL write failed for insert id=" + id);
                // In production: fail the request or use a dead-letter queue
                // Here: log and continue (acceptable for a demo/project)
            }
        }

        // Update in-memory indexes
        store.put(id, item);
        bruteForce.insert(item);

        kdtLock.writeLock().lock();
        try { kdTree.insert(item); }
        finally { kdtLock.writeLock().unlock(); }

        hnsw.insert(item, metric);

        return item;
    }

    // ── Delete ───────────────────────────────────────────────────────────

    public boolean delete(int id) {
        if (!store.containsKey(id)) return false;

        // WAL FIRST
        if (wal != null) {
            try { wal.logDelete(id); }
            catch (IOException e) { log.warning("WAL write failed for delete id=" + id); }
        }

        store.remove(id);
        bruteForce.remove(id);
        hnsw.remove(id);

        // KDTree requires full rebuild on delete (limitation of the data structure)
        kdtLock.writeLock().lock();
        try { kdTree.rebuild(new ArrayList<>(store.values())); }
        finally { kdtLock.writeLock().unlock(); }

        return true;
    }

    // ── Search ───────────────────────────────────────────────────────────

    public SearchOutput search(float[] query, int k, String metricName, String algo) {
        DistanceMetric metric = DistanceMetric.fromName(metricName);
        long start = System.nanoTime();

        List<SearchResult> raw = switch (algo.toLowerCase()) {
            case "bruteforce" -> bruteForce.knn(query, k, metric);
            case "kdtree"     -> {
                kdtLock.readLock().lock();
                try { yield kdTree.knn(query, k, metric); }
                finally { kdtLock.readLock().unlock(); }
            }
            default           -> hnsw.knn(query, k, 50, metric);
        };

        long latencyUs = (System.nanoTime() - start) / 1000;

        // Hydrate results with full item data
        List<SearchResult> hydrated = new ArrayList<>();
        for (SearchResult r : raw) {
            VectorItem item = store.get(r.getId());
            if (item != null) { r.setItem(item); hydrated.add(r); }
        }

        return new SearchOutput(hydrated, latencyUs, algo, metricName);
    }

    // ── Benchmark ────────────────────────────────────────────────────────

    /**
     * Run all three algorithms on the same query and return timing.
     * This is the key differentiating feature for the UI and for interviews.
     */
    public BenchmarkOutput benchmark(float[] query, int k, String metricName) {
        DistanceMetric metric = DistanceMetric.fromName(metricName);

        long bfStart = System.nanoTime();
        bruteForce.knn(query, k, metric);
        long bfUs = (System.nanoTime() - bfStart) / 1000;

        long kdStart = System.nanoTime();
        kdtLock.readLock().lock();
        try { kdTree.knn(query, k, metric); }
        finally { kdtLock.readLock().unlock(); }
        long kdUs = (System.nanoTime() - kdStart) / 1000;

        long hnswStart = System.nanoTime();
        hnsw.knn(query, k, 50, metric);
        long hnswUs = (System.nanoTime() - hnswStart) / 1000;

        return new BenchmarkOutput(bfUs, kdUs, hnswUs, store.size());
    }

    // ── Recall Test (FAANG-level feature) ───────────────────────────────

    /**
     * Compute recall@k: what fraction of the brute-force top-k does HNSW return?
     *
     * recall@k = |HNSW results ∩ BruteForce results| / k
     *
     * This is the standard metric used by the ann-benchmarks project
     * (https://ann-benchmarks.com) to evaluate vector indexes.
     * A recall@10 of 0.95+ is production-grade.
     *
     * Being able to explain and compute this in an interview is elite-level.
     */
    public double computeRecall(float[] query, int k) {
        DistanceMetric metric = DistanceMetric.cosine();
        List<SearchResult> groundTruth = bruteForce.knn(query, k, metric);
        List<SearchResult> hnswResult  = hnsw.knn(query, k, 50, metric);

        Set<Integer> gtIds = new HashSet<>();
        for (SearchResult r : groundTruth) gtIds.add(r.getId());

        long matches = hnswResult.stream().filter(r -> gtIds.contains(r.getId())).count();
        return (double) matches / Math.max(1, groundTruth.size());
    }

    // ── Accessors ────────────────────────────────────────────────────────

    public Collection<VectorItem> getAll()        { return Collections.unmodifiableCollection(store.values()); }
    public int                    size()           { return store.size(); }
    public int                    getDims()        { return dims; }
    public HNSW.GraphInfo         getHnswInfo()    { return hnsw.getInfo(); }

    // ── Output DTOs ───────────────────────────────────────────────────────

    public record SearchOutput(List<SearchResult> results, long latencyUs,
                               String algo, String metric) {}

    public record BenchmarkOutput(long bruteForceUs, long kdTreeUs, long hnswUs, int itemCount) {}
}
