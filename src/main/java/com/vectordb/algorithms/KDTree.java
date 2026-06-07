package com.vectordb.algorithms;

import com.vectordb.model.SearchResult;
import com.vectordb.model.VectorItem;
import com.vectordb.util.DistanceMetric;

import java.util.*;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * KD-Tree — Binary space partitioning for exact k-NN in low dimensions.
 *
 * Complexity:
 *   Insert:  O(log N) average, O(N) worst case (unbalanced)
 *   Search:  O(log N) average — degrades toward O(N) as dims → ∞
 *
 * Known limitation — Curse of Dimensionality:
 *   KD-Tree pruning relies on axis-aligned bounding hypercubes.
 *   In high dimensions (d >> 20), nearly all space is near the boundary
 *   of each hypersphere, so almost no subtrees get pruned.
 *   At 768D (Ollama embeddings), this becomes close to brute force.
 *   → HNSW wins there. KD-Tree wins at 16D demo vectors.
 *
 * Thread safety: ReentrantReadWriteLock — multiple concurrent readers,
 * exclusive writer. This is the correct primitive for a read-heavy index.
 */
public class KDTree {

    private static final class Node {
        final VectorItem item;
        Node left, right;
        Node(VectorItem item) { this.item = item; }
    }

    private Node root;
    private final int dims;
    private final ReadWriteLock lock = new ReentrantReadWriteLock();

    public KDTree(int dims) {
        this.dims = dims;
    }

    // ── Insert ──────────────────────────────────────────────────────────

    public void insert(VectorItem item) {
        lock.writeLock().lock();
        try {
            root = insertRec(root, item, 0);
        } finally {
            lock.writeLock().unlock();
        }
    }

    private Node insertRec(Node node, VectorItem item, int depth) {
        if (node == null) return new Node(item);
        int axis = depth % dims;
        if (item.getEmbedding()[axis] < node.item.getEmbedding()[axis]) {
            node.left  = insertRec(node.left,  item, depth + 1);
        } else {
            node.right = insertRec(node.right, item, depth + 1);
        }
        return node;
    }

    // ── Search ──────────────────────────────────────────────────────────

    /**
     * k-NN search using the ball-within-hyperslab pruning rule.
     * If the closest possible point in a subtree can't beat the k-th
     * best found so far, prune the entire subtree.
     */
    public List<SearchResult> knn(float[] query, int k, DistanceMetric metric) {
        lock.readLock().lock();
        try {
            // Max-heap of size k: we want the k smallest distances.
            // PriorityQueue is a min-heap by default; reverse for max-heap.
            PriorityQueue<SearchResult> heap =
                new PriorityQueue<>(k, (a, b) -> Float.compare(b.getDistance(), a.getDistance()));

            knnRec(root, query, k, 0, metric, heap);

            List<SearchResult> result = new ArrayList<>(heap);
            result.sort(Comparator.naturalOrder());
            return result;
        } finally {
            lock.readLock().unlock();
        }
    }

    private void knnRec(Node node, float[] query, int k, int depth,
                        DistanceMetric metric, PriorityQueue<SearchResult> heap) {
        if (node == null) return;

        float dist = metric.compute(query, node.item.getEmbedding());

        // Add to heap if we have room or this is closer than the worst so far
        if (heap.size() < k || dist < heap.peek().getDistance()) {
            heap.offer(new SearchResult(node.item.getId(), dist));
            if (heap.size() > k) heap.poll(); // evict the worst
        }

        int   axis = depth % dims;
        float diff = query[axis] - node.item.getEmbedding()[axis];

        // Always search the closer subtree first
        Node closer  = diff < 0 ? node.left  : node.right;
        Node farther = diff < 0 ? node.right : node.left;

        knnRec(closer, query, k, depth + 1, metric, heap);

        // Prune farther subtree: if the axis gap alone exceeds our worst match, skip
        if (heap.size() < k || Math.abs(diff) < heap.peek().getDistance()) {
            knnRec(farther, query, k, depth + 1, metric, heap);
        }
    }

    // ── Rebuild (needed after delete) ────────────────────────────────────

    public void rebuild(List<VectorItem> items) {
        lock.writeLock().lock();
        try {
            root = null;
            for (VectorItem item : items) root = insertRec(root, item, 0);
        } finally {
            lock.writeLock().unlock();
        }
    }
}
