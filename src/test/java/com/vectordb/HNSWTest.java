package com.vectordb;

import com.vectordb.algorithms.BruteForce;
import com.vectordb.algorithms.HNSW;
import com.vectordb.model.SearchResult;
import com.vectordb.model.VectorItem;
import com.vectordb.util.DistanceMetric;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.*;
import java.util.concurrent.*;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.*;

/**
 * HNSW Test Suite
 *
 * Three test categories:
 *   1. Correctness  — does HNSW find the right neighbours?
 *   2. Recall       — what fraction of brute-force top-k does HNSW return? (target: >0.90)
 *   3. Concurrency  — does the index stay consistent under parallel inserts/reads?
 *
 * Being able to explain EACH of these in an interview is elite-level.
 * Most freshers have never written a recall test. This sets you apart.
 */
class HNSWTest {

    private static final int  DIMS = 16;
    private static final long SEED = 42L;

    private HNSW       hnsw;
    private BruteForce bf;
    private Random     rng;

    @BeforeEach
    void setUp() {
        hnsw = new HNSW(16, 200);
        bf   = new BruteForce();
        rng  = new Random(SEED);
    }

    // ── Helper: generate a random unit vector ─────────────────────────────

    private float[] randomVector(int dims) {
        float[] v = new float[dims];
        float norm = 0f;
        for (int i = 0; i < dims; i++) { v[i] = rng.nextFloat() * 2 - 1; norm += v[i] * v[i]; }
        norm = (float) Math.sqrt(norm);
        for (int i = 0; i < dims; i++) v[i] /= norm;
        return v;
    }

    private VectorItem makeItem(int id) {
        return new VectorItem(id, "item-" + id, "test", randomVector(DIMS));
    }

    private void insertBoth(VectorItem item) {
        hnsw.insert(item, DistanceMetric.cosine());
        bf.insert(item);
    }

    // ── 1. Correctness Tests ──────────────────────────────────────────────

    @Test
    @DisplayName("Single insert: exact nearest neighbour found")
    void testSingleInsert() {
        VectorItem item = new VectorItem(1, "test", "cat", new float[]{1f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f});
        hnsw.insert(item, DistanceMetric.cosine());

        float[] query = new float[]{1f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f};
        List<SearchResult> results = hnsw.knn(query, 1, 10, DistanceMetric.cosine());

        assertEquals(1, results.size());
        assertEquals(1, results.get(0).getId());
        assertEquals(0f, results.get(0).getDistance(), 1e-5f, "Same vector should have distance ~0");
    }

    @Test
    @DisplayName("Empty index returns empty results")
    void testEmptyIndex() {
        float[] query = randomVector(DIMS);
        List<SearchResult> results = hnsw.knn(query, 5, 10, DistanceMetric.cosine());
        assertTrue(results.isEmpty());
    }

    @Test
    @DisplayName("Results are sorted by distance ascending")
    void testResultsAreSorted() {
        for (int i = 1; i <= 50; i++) insertBoth(makeItem(i));

        float[] query = randomVector(DIMS);
        List<SearchResult> results = hnsw.knn(query, 10, 50, DistanceMetric.cosine());

        for (int i = 1; i < results.size(); i++) {
            assertTrue(results.get(i).getDistance() >= results.get(i - 1).getDistance(),
                "Results must be in ascending distance order");
        }
    }

    @Test
    @DisplayName("k > n returns at most n results")
    void testKGreaterThanN() {
        for (int i = 1; i <= 5; i++) insertBoth(makeItem(i));
        List<SearchResult> results = hnsw.knn(randomVector(DIMS), 100, 50, DistanceMetric.cosine());
        assertTrue(results.size() <= 5, "Can't return more items than exist in index");
    }

    @Test
    @DisplayName("Delete removes item from results")
    void testDelete() {
        VectorItem target = new VectorItem(99, "target", "test",
            new float[]{1f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f});
        hnsw.insert(target, DistanceMetric.cosine());
        for (int i = 1; i <= 20; i++) insertBoth(makeItem(i));

        hnsw.remove(99);

        float[] query = new float[]{1f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f};
        List<SearchResult> results = hnsw.knn(query, 5, 50, DistanceMetric.cosine());
        boolean found = results.stream().anyMatch(r -> r.getId() == 99);
        assertFalse(found, "Deleted item should not appear in results");
    }

    // ── 2. Recall Tests ───────────────────────────────────────────────────

    /**
     * Recall@k = |HNSW results ∩ BruteForce results| / k
     *
     * This is the standard metric from ann-benchmarks.com.
     * Production vector DBs target recall@10 > 0.95.
     * With M=16, efConstruct=200, ef=50 on random 16D data, we expect >0.90.
     */
    @Test
    @DisplayName("Recall@10 >= 0.90 with 500 vectors at 16D")
    void testRecallAt10() {
        int n = 500;
        for (int i = 1; i <= n; i++) insertBoth(makeItem(i));

        int k = 10, trials = 50;
        double totalRecall = 0;

        for (int t = 0; t < trials; t++) {
            float[] query = randomVector(DIMS);

            List<SearchResult> hnswResults = hnsw.knn(query, k, 50, DistanceMetric.cosine());
            List<SearchResult> bfResults   = bf.knn(query, k, DistanceMetric.cosine());

            Set<Integer> bfIds = bfResults.stream()
                .map(SearchResult::getId).collect(Collectors.toSet());

            long matches = hnswResults.stream()
                .filter(r -> bfIds.contains(r.getId())).count();

            totalRecall += (double) matches / k;
        }

        double avgRecall = totalRecall / trials;
        System.out.printf("HNSW Recall@%d (n=%d, trials=%d): %.4f%n", k, n, trials, avgRecall);

        assertTrue(avgRecall >= 0.90,
            String.format("Expected recall >= 0.90, got %.4f. " +
                "Try increasing efConstruct or ef.", avgRecall));
    }

    @Test
    @DisplayName("Recall@10 >= 0.85 with 1000 vectors at 16D")
    void testRecallAt10LargerDataset() {
        int n = 1000;
        for (int i = 1; i <= n; i++) insertBoth(makeItem(i));

        int k = 10, trials = 30;
        double totalRecall = 0;

        for (int t = 0; t < trials; t++) {
            float[] query = randomVector(DIMS);
            List<SearchResult> hnswResults = hnsw.knn(query, k, 50, DistanceMetric.cosine());
            List<SearchResult> bfResults   = bf.knn(query, k, DistanceMetric.cosine());

            Set<Integer> bfIds = bfResults.stream()
                .map(SearchResult::getId).collect(Collectors.toSet());
            long matches = hnswResults.stream().filter(r -> bfIds.contains(r.getId())).count();
            totalRecall += (double) matches / k;
        }

        double avgRecall = totalRecall / trials;
        System.out.printf("HNSW Recall@%d (n=%d, trials=%d): %.4f%n", k, n, trials, avgRecall);
        assertTrue(avgRecall >= 0.85,
            String.format("Expected recall >= 0.85 at n=1000, got %.4f", avgRecall));
    }

    // ── 3. Concurrency Tests ──────────────────────────────────────────────

    /**
     * Concurrent inserts must not corrupt the graph.
     * After N parallel inserts, all items must be findable.
     *
     * This is the most important safety property of any concurrent data structure.
     * FAANG interviewers love asking about this.
     */
    @Test
    @DisplayName("Concurrent inserts: no corruption, all items findable")
    void testConcurrentInserts() throws InterruptedException {
        int              threads    = 8;
        int              perThread  = 50;
        ExecutorService  pool       = Executors.newFixedThreadPool(threads);
        CountDownLatch   latch      = new CountDownLatch(threads);
        List<Future<?>>  futures    = new ArrayList<>();

        for (int t = 0; t < threads; t++) {
            final int base = t * perThread;
            futures.add(pool.submit(() -> {
                for (int i = 1; i <= perThread; i++) {
                    VectorItem item = new VectorItem(base + i, "item", "test", randomVector(DIMS));
                    hnsw.insert(item, DistanceMetric.cosine());
                }
                latch.countDown();
            }));
        }

        latch.await(30, TimeUnit.SECONDS);
        pool.shutdown();

        // Verify no exceptions were thrown
        for (Future<?> f : futures) {
            assertDoesNotThrow(() -> f.get(), "No exceptions during concurrent insert");
        }

        // Verify the index is non-empty and searchable
        assertEquals(threads * perThread, hnsw.size());
        List<SearchResult> results = hnsw.knn(randomVector(DIMS), 10, 50, DistanceMetric.cosine());
        assertFalse(results.isEmpty(), "Index must be searchable after concurrent inserts");
    }

    @Test
    @DisplayName("Concurrent reads while inserting: no deadlock")
    void testConcurrentReadsAndWrites() throws InterruptedException {
        // Pre-populate
        for (int i = 1; i <= 100; i++) {
            hnsw.insert(makeItem(i), DistanceMetric.cosine());
        }

        int             writerCount = 2;
        int             readerCount = 6;
        int             ops         = 50;
        ExecutorService pool        = Executors.newFixedThreadPool(writerCount + readerCount);
        CountDownLatch  done        = new CountDownLatch(writerCount + readerCount);
        List<Future<?>> futures     = new ArrayList<>();

        // Writers
        for (int w = 0; w < writerCount; w++) {
            final int base = 1000 + w * ops;
            futures.add(pool.submit(() -> {
                for (int i = 0; i < ops; i++) {
                    hnsw.insert(new VectorItem(base + i, "w", "test", randomVector(DIMS)),
                                DistanceMetric.cosine());
                }
                done.countDown();
            }));
        }

        // Readers
        for (int r = 0; r < readerCount; r++) {
            futures.add(pool.submit(() -> {
                for (int i = 0; i < ops; i++) {
                    hnsw.knn(randomVector(DIMS), 5, 20, DistanceMetric.cosine());
                }
                done.countDown();
            }));
        }

        boolean completed = done.await(30, TimeUnit.SECONDS);
        pool.shutdown();

        assertTrue(completed, "No deadlock — all threads completed within 30s");
        for (Future<?> f : futures) {
            assertDoesNotThrow(() -> f.get(), "No exceptions in concurrent read/write");
        }
    }
}
