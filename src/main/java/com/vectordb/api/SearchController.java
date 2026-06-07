package com.vectordb.api;

import com.vectordb.algorithms.HNSW;
import com.vectordb.core.VectorDB;
import com.vectordb.model.SearchResult;
import com.vectordb.model.VectorItem;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.*;
import java.util.stream.Collectors;

/**
 * REST API — Demo vector search endpoints.
 *
 * Spring Boot maps these annotations to HTTP routes automatically.
 * If you know Express.js, this maps 1:1:
 *   @RestController       = express.Router()
 *   @GetMapping("/path")  = router.get('/path', handler)
 *   @RequestParam         = req.query.param
 *   @PathVariable         = req.params.id
 *   ResponseEntity        = res.status(200).json(...)
 */
@RestController
@RequestMapping("/api")
@CrossOrigin(origins = "*")  // CORS — allow the frontend to call the API
public class SearchController {

    private final VectorDB db;

    public SearchController(VectorDB db) {
        this.db = db;
    }

    // GET /search?v=f1,f2,...&k=5&metric=cosine&algo=hnsw
    @GetMapping("/search")
    public ResponseEntity<?> search(
        @RequestParam String v,
        @RequestParam(defaultValue = "5") int k,
        @RequestParam(defaultValue = "cosine") String metric,
        @RequestParam(defaultValue = "hnsw") String algo
    ) {
        float[] query = parseVector(v);
        if (query.length != db.getDims()) {
            return ResponseEntity.badRequest()
                .body(Map.of("error", "Need " + db.getDims() + "D vector, got " + query.length + "D"));
        }

        VectorDB.SearchOutput out = db.search(query, k, metric, algo);

        List<Map<String, Object>> results = out.results().stream()
            .map(r -> {
                Map<String, Object> m = new LinkedHashMap<>();
                m.put("id",        r.getId());
                m.put("metadata",  r.getItem().getMetadata());
                m.put("category",  r.getItem().getCategory());
                m.put("distance",  r.getDistance());
                m.put("embedding", toList(r.getItem().getEmbedding()));
                return m;
            })
            .collect(Collectors.toList());

        return ResponseEntity.ok(Map.of(
            "results",   results,
            "latencyUs", out.latencyUs(),
            "algo",      out.algo(),
            "metric",    out.metric()
        ));
    }

    // POST /insert  {"metadata":"...","category":"...","embedding":[...]}
    @PostMapping("/insert")
    public ResponseEntity<?> insert(@RequestBody Map<String, Object> body) {
        String metadata  = (String) body.getOrDefault("metadata", "");
        String category  = (String) body.getOrDefault("category", "");
        List<?> embList  = (List<?>) body.get("embedding");

        if (metadata.isBlank() || embList == null) {
            return ResponseEntity.badRequest().body(Map.of("error", "invalid body"));
        }

        float[] emb = new float[embList.size()];
        for (int i = 0; i < embList.size(); i++) emb[i] = ((Number) embList.get(i)).floatValue();

        if (emb.length != db.getDims()) {
            return ResponseEntity.badRequest()
                .body(Map.of("error", "Need " + db.getDims() + "D vector"));
        }

        VectorItem item = db.insert(metadata, category, emb, com.vectordb.util.DistanceMetric.cosine());
        return ResponseEntity.ok(Map.of("id", item.getId()));
    }

    // DELETE /delete/{id}
    @DeleteMapping("/delete/{id}")
    public ResponseEntity<?> delete(@PathVariable int id) {
        boolean ok = db.delete(id);
        return ResponseEntity.ok(Map.of("ok", ok));
    }

    // GET /items
    @GetMapping("/items")
    public ResponseEntity<?> items() {
        List<Map<String, Object>> result = db.getAll().stream()
            .map(v -> {
                Map<String, Object> m = new LinkedHashMap<>();
                m.put("id",        v.getId());
                m.put("metadata",  v.getMetadata());
                m.put("category",  v.getCategory());
                m.put("embedding", toList(v.getEmbedding()));
                return m;
            })
            .collect(Collectors.toList());
        return ResponseEntity.ok(result);
    }

    // GET /benchmark?v=...&k=5&metric=cosine
    @GetMapping("/benchmark")
    public ResponseEntity<?> benchmark(
        @RequestParam String v,
        @RequestParam(defaultValue = "5") int k,
        @RequestParam(defaultValue = "cosine") String metric
    ) {
        float[] query = parseVector(v);
        if (query.length != db.getDims()) {
            return ResponseEntity.badRequest().body(Map.of("error", "dimension mismatch"));
        }
        VectorDB.BenchmarkOutput b = db.benchmark(query, k, metric);
        return ResponseEntity.ok(Map.of(
            "bruteforceUs", b.bruteForceUs(),
            "kdtreeUs",     b.kdTreeUs(),
            "hnswUs",       b.hnswUs(),
            "itemCount",    b.itemCount()
        ));
    }

    // GET /hnsw-info
    @GetMapping("/hnsw-info")
    public ResponseEntity<?> hnswInfo() {
        HNSW.GraphInfo gi = db.getHnswInfo();
        Map<String, Object> resp = new LinkedHashMap<>();
        resp.put("topLayer",      gi.topLayer());
        resp.put("nodeCount",     gi.nodeCount());
        resp.put("nodesPerLayer", gi.nodesPerLayer());
        resp.put("edgesPerLayer", gi.edgesPerLayer());
        resp.put("nodes", gi.nodes().stream().map(n -> Map.of(
            "id",       n.id(),
            "metadata", n.metadata(),
            "category", n.category(),
            "maxLayer", n.maxLayer()
        )).collect(Collectors.toList()));
        resp.put("edges", gi.edges().stream().map(e -> Map.of(
            "src", e.src(), "dst", e.dst(), "layer", e.layer()
        )).collect(Collectors.toList()));
        return ResponseEntity.ok(resp);
    }

    // GET /recall?k=10  — compute HNSW recall against brute force ground truth
    @GetMapping("/recall")
    public ResponseEntity<?> recall(@RequestParam(defaultValue = "10") int k) {
        // Use the first stored item's embedding as a test query
        Optional<VectorItem> sample = db.getAll().stream().findFirst();
        if (sample.isEmpty()) return ResponseEntity.ok(Map.of("recall", 0.0, "message", "no data"));

        double recall = db.computeRecall(sample.get().getEmbedding(), k);
        return ResponseEntity.ok(Map.of(
            "recall_at_k", recall,
            "k",           k,
            "n",           db.size(),
            "description", "Fraction of brute-force top-k found by HNSW"
        ));
    }

    // GET /stats
    @GetMapping("/stats")
    public ResponseEntity<?> stats() {
        return ResponseEntity.ok(Map.of(
            "count",      db.size(),
            "dims",       db.getDims(),
            "algorithms", List.of("bruteforce", "kdtree", "hnsw"),
            "metrics",    List.of("euclidean", "cosine", "manhattan")
        ));
    }

    // ── Helpers ───────────────────────────────────────────────────────────

    private float[] parseVector(String s) {
        if (s == null || s.isBlank()) return new float[0];
        String[] parts = s.split(",");
        float[] v = new float[parts.length];
        try {
            for (int i = 0; i < parts.length; i++) v[i] = Float.parseFloat(parts[i].trim());
        } catch (NumberFormatException e) {
            return new float[0];
        }
        return v;
    }

    private List<Float> toList(float[] arr) {
        List<Float> list = new ArrayList<>(arr.length);
        for (float f : arr) list.add(f);
        return list;
    }
}
