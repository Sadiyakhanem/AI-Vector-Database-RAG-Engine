package com.vectordb.util;

/**
 * Distance / similarity metrics.
 *
 * All functions return a NON-NEGATIVE float where LOWER = MORE SIMILAR.
 * Cosine distance = 1 - cosine_similarity, so it sits in [0, 2].
 *
 * Why a functional interface instead of an enum?
 * → Cleaner lambda passing, easier to add custom metrics at runtime.
 */
@FunctionalInterface
public interface DistanceMetric {

    float compute(float[] a, float[] b);

    // ── Built-in metrics ────────────────────────────────────────────────

    static DistanceMetric euclidean() {
        return (a, b) -> {
            float sum = 0f;
            for (int i = 0; i < a.length; i++) {
                float d = a[i] - b[i];
                sum += d * d;
            }
            return (float) Math.sqrt(sum);
        };
    }

    static DistanceMetric cosine() {
        return (a, b) -> {
            float dot = 0f, normA = 0f, normB = 0f;
            for (int i = 0; i < a.length; i++) {
                dot   += a[i] * b[i];
                normA += a[i] * a[i];
                normB += b[i] * b[i];
            }
            if (normA < 1e-9f || normB < 1e-9f) return 1f;
            return 1f - dot / (float)(Math.sqrt(normA) * Math.sqrt(normB));
        };
    }

    static DistanceMetric manhattan() {
        return (a, b) -> {
            float sum = 0f;
            for (int i = 0; i < a.length; i++) sum += Math.abs(a[i] - b[i]);
            return sum;
        };
    }

    static DistanceMetric fromName(String name) {
        return switch (name == null ? "" : name.toLowerCase()) {
            case "cosine"    -> cosine();
            case "manhattan" -> manhattan();
            default          -> euclidean();
        };
    }
}
