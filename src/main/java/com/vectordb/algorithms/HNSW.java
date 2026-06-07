package com.vectordb.algorithms;

import com.vectordb.model.SearchResult;
import com.vectordb.model.VectorItem;
import com.vectordb.util.DistanceMetric;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * HNSW — Hierarchical Navigable Small World Graph
 *
 * This is the same algorithm used by Pinecone, Weaviate, Chroma, and Milvus.
 *
 * Key idea:
 *   Build a multi-layer graph. Layer 0 has all nodes with dense connections.
 *   Higher layers have exponentially fewer nodes with long-range connections.
 *   Search starts at the top (sparse, fast navigation) and zooms in at layer 0.
 *
 * Complexity:
 *   Insert:  O(log N) expected
 *   Search:  O(log N) expected
 *   Space:   O(N · M) where M is the max connections per node
 *
 * Parameters:
 *   M            = max connections per node per layer (default 16)
 *                  Higher M → better recall, more memory, slower insert
 *   efConstruct  = beam width during construction (default 200)
 *                  Higher → better graph quality, slower inserts
 *   efSearch     = beam width during search (default 50)
 *                  Higher → better recall, slower queries
 *                  Trade-off: recall vs latency
 *
 * Thread safety:
 *   ReentrantReadWriteLock on the entire graph.
 *   Production improvement: per-node locking for higher write concurrency.
 *   For a fresher project, global RW lock is correct and explainable.
 */
public class HNSW {

    // ── Internal graph node ──────────────────────────────────────────────
    private static final class Node {
        final VectorItem          item;
        final int                 maxLayer;
        final List<List<Integer>> neighbors; // neighbors[layer] = list of neighbor ids

        Node(VectorItem item, int maxLayer) {
            this.item      = item;
            this.maxLayer  = maxLayer;
            this.neighbors = new ArrayList<>(maxLayer + 1);
            for (int i = 0; i <= maxLayer; i++) neighbors.add(new ArrayList<>());
        }
    }

    // ── Graph state ──────────────────────────────────────────────────────
    private final ConcurrentHashMap<Integer, Node> graph = new ConcurrentHashMap<>();
    private final ReadWriteLock lock     = new ReentrantReadWriteLock();

    private volatile int entryPoint = -1;
    private volatile int topLayer   = -1;

    // ── Hyperparameters ───────────────────────────────────────────────────
    private final int   M;          // max neighbors per layer (except layer 0)
    private final int   M0;         // max neighbors at layer 0 = 2*M
    private final int   efConstruct;
    private final float mL;         // level generation factor = 1 / ln(M)
    private final Random rng = new Random(42); // fixed seed for reproducibility

    public HNSW(int m, int efConstruct) {
        this.M           = m;
        this.M0          = 2 * m;
        this.efConstruct = efConstruct;
        this.mL          = 1.0f / (float) Math.log(m);
    }

    // ── Level generation ─────────────────────────────────────────────────

    /**
     * Random level assignment using an exponential distribution.
     * mL = 1/ln(M) ensures the expected number of nodes at layer l is N * e^(-l/mL),
     * giving an exponentially sparser graph at higher layers.
     */
    private int randomLevel() {
        return (int) Math.floor(-Math.log(rng.nextDouble()) * mL);
    }

    // ── Insert ───────────────────────────────────────────────────────────

    public void insert(VectorItem item, DistanceMetric metric) {
        lock.writeLock().lock();
        try {
            int level = randomLevel();
            Node newNode = new Node(item, level);
            graph.put(item.getId(), newNode);

            // First node — becomes the entry point
            if (entryPoint == -1) {
                entryPoint = item.getId();
                topLayer   = level;
                return;
            }

            int ep = entryPoint;

            // Phase 1: Greedy descent from topLayer down to level+1
            // (we only need ef=1 here — just finding the neighborhood)
            for (int lc = topLayer; lc > level; lc--) {
                List<SearchResult> w = searchLayer(item.getEmbedding(), ep, 1, lc, metric);
                if (!w.isEmpty()) ep = w.get(0).getId();
            }

            // Phase 2: Beam search from min(topLayer, level) down to layer 0
            // At each layer, connect the new node to its M nearest neighbors
            for (int lc = Math.min(topLayer, level); lc >= 0; lc--) {
                int maxM = (lc == 0) ? M0 : M;
                List<SearchResult> w = searchLayer(item.getEmbedding(), ep, efConstruct, lc, metric);

                // Select the best maxM candidates
                List<Integer> selected = selectNeighbors(w, maxM);
                newNode.neighbors.get(lc).addAll(selected);

                // Bidirectional connection: also add this node to each neighbor's list
                for (int neighborId : selected) {
                    Node neighbor = graph.get(neighborId);
                    if (neighbor == null) continue;
                    if (lc >= neighbor.neighbors.size()) continue;

                    List<Integer> conn = neighbor.neighbors.get(lc);
                    conn.add(item.getId());

                    // Prune if neighbor now has too many connections
                    if (conn.size() > maxM) {
                        pruneConnections(conn, neighbor.item.getEmbedding(), maxM, metric);
                    }
                }

                if (!w.isEmpty()) ep = w.get(0).getId();
            }

            // Update entry point if this node reaches a higher layer
            if (level > topLayer) {
                topLayer   = level;
                entryPoint = item.getId();
            }
        } finally {
            lock.writeLock().unlock();
        }
    }

    // ── Search ───────────────────────────────────────────────────────────

    public List<SearchResult> knn(float[] query, int k, int ef, DistanceMetric metric) {
        lock.readLock().lock();
        try {
            if (entryPoint == -1) return Collections.emptyList();

            int ep = entryPoint;

            // Greedy descent to layer 1 with ef=1 (fast navigation)
            for (int lc = topLayer; lc > 0; lc--) {
                List<SearchResult> w = searchLayer(query, ep, 1, lc, metric);
                if (!w.isEmpty()) ep = w.get(0).getId();
            }

            // Full beam search at layer 0 with ef=max(ef, k)
            List<SearchResult> w = searchLayer(query, ep, Math.max(ef, k), 0, metric);

            if (w.size() > k) w = w.subList(0, k);
            return w;
        } finally {
            lock.readLock().unlock();
        }
    }

    // ── Core beam search within a single layer ───────────────────────────

    /**
     * Greedy beam search on one HNSW layer.
     *
     * Uses two priority queues:
     *   candidates — min-heap by distance (explore closest first)
     *   found      — max-heap of size ef (keep the ef closest found so far)
     *
     * Termination: stop when the closest unexplored candidate is farther
     * than the worst element in `found` — we can't improve anymore.
     */
    private List<SearchResult> searchLayer(float[] query, int entryId,
                                           int ef, int layer, DistanceMetric metric) {
        Set<Integer> visited = new HashSet<>();

        // Min-heap: explore closest candidates first
        PriorityQueue<SearchResult> candidates =
            new PriorityQueue<>(Comparator.comparingDouble(SearchResult::getDistance));

        // Max-heap of size ef: tracks the ef best found so far
        PriorityQueue<SearchResult> found =
            new PriorityQueue<>((a, b) -> Float.compare(b.getDistance(), a.getDistance()));

        Node entry = graph.get(entryId);
        if (entry == null) return Collections.emptyList();

        float d0 = metric.compute(query, entry.item.getEmbedding());
        visited.add(entryId);
        candidates.offer(new SearchResult(entryId, d0));
        found.offer(new SearchResult(entryId, d0));

        while (!candidates.isEmpty()) {
            SearchResult current = candidates.poll();

            // Early termination: can't do better than what we have
            if (!found.isEmpty() && found.size() >= ef
                    && current.getDistance() > found.peek().getDistance()) {
                break;
            }

            Node currentNode = graph.get(current.getId());
            if (currentNode == null || layer >= currentNode.neighbors.size()) continue;

            for (int neighborId : currentNode.neighbors.get(layer)) {
                if (visited.contains(neighborId)) continue;
                visited.add(neighborId);

                Node neighbor = graph.get(neighborId);
                if (neighbor == null) continue;

                float nd = metric.compute(query, neighbor.item.getEmbedding());

                if (found.size() < ef || nd < found.peek().getDistance()) {
                    candidates.offer(new SearchResult(neighborId, nd));
                    found.offer(new SearchResult(neighborId, nd));
                    if (found.size() > ef) found.poll();
                }
            }
        }

        List<SearchResult> result = new ArrayList<>(found);
        result.sort(Comparator.naturalOrder());
        return result;
    }

    // ── Neighbor selection & pruning ─────────────────────────────────────

    private List<Integer> selectNeighbors(List<SearchResult> candidates, int maxM) {
        List<Integer> selected = new ArrayList<>();
        for (int i = 0; i < Math.min(candidates.size(), maxM); i++) {
            selected.add(candidates.get(i).getId());
        }
        return selected;
    }

    /**
     * Prune a connection list to maxM by keeping the closest neighbors.
     * Modifies the list in-place.
     */
    private void pruneConnections(List<Integer> connections, float[] origin,
                                  int maxM, DistanceMetric metric) {
        List<SearchResult> scored = new ArrayList<>();
        for (int id : connections) {
            Node n = graph.get(id);
            if (n != null) scored.add(new SearchResult(id, metric.compute(origin, n.item.getEmbedding())));
        }
        scored.sort(Comparator.naturalOrder());
        connections.clear();
        for (int i = 0; i < Math.min(scored.size(), maxM); i++) {
            connections.add(scored.get(i).getId());
        }
    }

    // ── Delete ───────────────────────────────────────────────────────────

    public void remove(int id) {
        lock.writeLock().lock();
        try {
            if (!graph.containsKey(id)) return;

            // Remove this id from all neighbor lists across all nodes
            for (Node node : graph.values()) {
                for (List<Integer> layerNeighbors : node.neighbors) {
                    layerNeighbors.remove(Integer.valueOf(id));
                }
            }

            // If we're deleting the entry point, pick a new one
            if (entryPoint == id) {
                entryPoint = graph.keySet().stream()
                    .filter(k -> k != id)
                    .findFirst()
                    .orElse(-1);
            }

            graph.remove(id);
        } finally {
            lock.writeLock().unlock();
        }
    }

    // ── Graph metadata (for /hnsw-info endpoint) ─────────────────────────

    public GraphInfo getInfo() {
        lock.readLock().lock();
        try {
            int maxL = Math.max(topLayer + 1, 1);
            int[] nodesPerLayer = new int[maxL];
            int[] edgesPerLayer = new int[maxL];
            List<NodeInfo> nodes = new ArrayList<>();
            List<EdgeInfo> edges = new ArrayList<>();

            for (Map.Entry<Integer, Node> entry : graph.entrySet()) {
                int  id   = entry.getKey();
                Node node = entry.getValue();
                nodes.add(new NodeInfo(id, node.item.getMetadata(),
                                       node.item.getCategory(), node.maxLayer));

                for (int lc = 0; lc <= Math.min(node.maxLayer, maxL - 1); lc++) {
                    nodesPerLayer[lc]++;
                    if (lc < node.neighbors.size()) {
                        for (int nid : node.neighbors.get(lc)) {
                            if (id < nid) { // avoid double-counting
                                edgesPerLayer[lc]++;
                                edges.add(new EdgeInfo(id, nid, lc));
                            }
                        }
                    }
                }
            }
            return new GraphInfo(topLayer, graph.size(), nodesPerLayer, edgesPerLayer, nodes, edges);
        } finally {
            lock.readLock().unlock();
        }
    }

    public int size() { return graph.size(); }

    // ── DTO inner classes ─────────────────────────────────────────────────
    public record GraphInfo(int topLayer, int nodeCount, int[] nodesPerLayer,
                            int[] edgesPerLayer, List<NodeInfo> nodes, List<EdgeInfo> edges) {}
    public record NodeInfo(int id, String metadata, String category, int maxLayer) {}
    public record EdgeInfo(int src, int dst, int layer) {}
}
