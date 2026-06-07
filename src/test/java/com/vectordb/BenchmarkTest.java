package com.vectordb;

import com.vectordb.algorithms.BruteForce;
import com.vectordb.algorithms.HNSW;
import com.vectordb.algorithms.KDTree;
import com.vectordb.model.SearchResult;
import com.vectordb.model.VectorItem;
import com.vectordb.util.DistanceMetric;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.*;

/**
 * Algorithm Benchmark — Generates the real numbers for your resume.
 *
 * Run this test with:
 *   mvn test -Dtest=BenchmarkTest -pl . 2>&1 | grep -A 20 "BENCHMARK"
 *
 * The output gives you concrete claims like:
 *   "HNSW is 43x faster than brute force at 10,000 vectors"
 *   "Recall@10 = 96.2% — production-grade accuracy"
 *
 * This is exactly how ann-benchmarks.com evaluates vector indexes.
 *
 * NOTE: This is a manual benchmark (not JMH). For a production-grade
 * benchmark with warmup, GC pauses handled, and statistical confidence
 * intervals, use JMH. This is sufficient for a fresher project.
 */
class BenchmarkTest {

    private static final int  DIMS = 16;
    private static final long SEED = 42L;

    private float[] randomVector(Random rng) {
        float[] v = new float[DIMS];
        for (int i = 0; i < DIMS; i++) v[i] = rng.nextFloat();
        return v;
    }

    @Test
    @DisplayName("Speed benchmark: HNSW vs KDTree vs BruteForce across dataset sizes")
    void benchmarkSpeed() {
        int[] sizes = {100, 500, 1_000, 5_000, 10_000};
        int   k     = 10;
        int   warmupRuns = 5;
        int   benchRuns  = 20;

        System.out.println("\n" + "=".repeat(70));
        System.out.println("BENCHMARK: HNSW vs KD-Tree vs BruteForce");
        System.out.printf("%-10s %-15s %-15s %-15s %-12s%n",
            "N", "BruteForce(μs)", "KDTree(μs)", "HNSW(μs)", "Speedup");
        System.out.println("-".repeat(70));

        for (int n : sizes) {
            Random     rng = new Random(SEED);
            BruteForce bf  = new BruteForce();
            KDTree     kdt = new KDTree(DIMS);
            HNSW       hnsw= new HNSW(16, 200);
            DistanceMetric cosine = DistanceMetric.cosine();

            // Insert N items
            for (int i = 1; i <= n; i++) {
                VectorItem item = new VectorItem(i, "item-" + i, "test", randomVector(rng));
                bf.insert(item);
                kdt.insert(item);
                hnsw.insert(item, cosine);
            }

            float[] query = randomVector(rng);

            // Warmup
            for (int w = 0; w < warmupRuns; w++) {
                bf.knn(query, k, cosine);
                kdt.knn(query, k, cosine);
                hnsw.knn(query, k, 50, cosine);
            }

            // Benchmark BruteForce
            long bfTotal = 0;
            for (int r = 0; r < benchRuns; r++) {
                long t = System.nanoTime();
                bf.knn(query, k, cosine);
                bfTotal += System.nanoTime() - t;
            }
            long bfAvgUs = (bfTotal / benchRuns) / 1000;

            // Benchmark KDTree
            long kdTotal = 0;
            for (int r = 0; r < benchRuns; r++) {
                long t = System.nanoTime();
                kdt.knn(query, k, cosine);
                kdTotal += System.nanoTime() - t;
            }
            long kdAvgUs = (kdTotal / benchRuns) / 1000;

            // Benchmark HNSW
            long hnswTotal = 0;
            for (int r = 0; r < benchRuns; r++) {
                long t = System.nanoTime();
                hnsw.knn(query, k, 50, cosine);
                hnswTotal += System.nanoTime() - t;
            }
            long hnswAvgUs = (hnswTotal / benchRuns) / 1000;

            double speedup = hnswAvgUs > 0 ? (double) bfAvgUs / hnswAvgUs : 0;

            System.out.printf("%-10d %-15d %-15d %-15d %-12.1fx%n",
                n, bfAvgUs, kdAvgUs, hnswAvgUs, speedup);
        }
        System.out.println("=".repeat(70));
    }

    @Test
    @DisplayName("Recall benchmark: HNSW accuracy vs brute force ground truth")
    void benchmarkRecall() {
        int[] sizes = {200, 500, 1_000, 2_000};
        int   k     = 10;
        int   trials = 50;

        System.out.println("\n" + "=".repeat(55));
        System.out.println("RECALL@" + k + " BENCHMARK: HNSW vs Ground Truth");
        System.out.printf("%-10s %-15s %-15s%n", "N", "Recall@" + k, "Status");
        System.out.println("-".repeat(55));

        for (int n : sizes) {
            Random     rng    = new Random(SEED);
            BruteForce bf     = new BruteForce();
            HNSW       hnsw   = new HNSW(16, 200);
            DistanceMetric cos = DistanceMetric.cosine();

            for (int i = 1; i <= n; i++) {
                VectorItem item = new VectorItem(i, "item-" + i, "test", randomVector(rng));
                bf.insert(item);
                hnsw.insert(item, cos);
            }

            double totalRecall = 0;
            for (int t = 0; t < trials; t++) {
                float[] q = randomVector(rng);
                List<SearchResult> hnswR = hnsw.knn(q, k, 50, cos);
                List<SearchResult> bfR   = bf.knn(q, k, cos);

                Set<Integer> bfIds = new HashSet<>();
                for (SearchResult r : bfR) bfIds.add(r.getId());
                long matches = hnswR.stream().filter(r -> bfIds.contains(r.getId())).count();
                totalRecall += (double) matches / k;
            }

            double recall = totalRecall / trials;
            String status = recall >= 0.95 ? "✓ EXCELLENT" : recall >= 0.90 ? "✓ GOOD" : "⚠ LOW";
            System.out.printf("%-10d %-15.4f %-15s%n", n, recall, status);
        }
        System.out.println("=".repeat(55));
        System.out.println("Target: recall@10 >= 0.95 (production-grade)");
        System.out.println();
    }
}
