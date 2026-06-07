package com.vectordb.api;

import com.vectordb.core.DocumentDB;
import com.vectordb.core.OllamaClient;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.*;
import java.util.stream.Collectors;

/**
 * REST API — Document embedding and RAG (Retrieval-Augmented Generation) endpoints.
 *
 * RAG Pipeline:
 *   1. User pastes text → we chunk it (250 words, 30 overlap)
 *   2. Each chunk → Ollama (nomic-embed-text) → 768D vector
 *   3. Vectors stored in HNSW DocumentDB
 *   4. User asks question → embed question → HNSW search → retrieve context
 *   5. Context + question → Ollama (llama3.2) → generated answer
 */
@RestController
@RequestMapping("/api/doc")
@CrossOrigin(origins = "*")
public class DocumentController {

    private final DocumentDB   docDB;
    private final OllamaClient ollama;

    public DocumentController(DocumentDB docDB, OllamaClient ollama) {
        this.docDB  = docDB;
        this.ollama = ollama;
    }

    // POST /doc/insert  {"title":"...","text":"..."}
    @PostMapping("/insert")
    public ResponseEntity<?> insert(@RequestBody Map<String, String> body) {
        String title = body.getOrDefault("title", "").trim();
        String text  = body.getOrDefault("text",  "").trim();

        if (title.isBlank() || text.isBlank()) {
            return ResponseEntity.badRequest().body(Map.of("error", "title and text required"));
        }

        List<String> chunks = DocumentDB.chunkText(text, 250, 30);
        List<Integer> ids   = new ArrayList<>();

        for (int i = 0; i < chunks.size(); i++) {
            float[] emb = ollama.embed(chunks.get(i));
            if (emb.length == 0) {
                return ResponseEntity.status(503).body(Map.of(
                    "error", "Ollama unavailable. Install from https://ollama.com " +
                             "then run: ollama pull nomic-embed-text && ollama pull llama3.2"
                ));
            }
            String chunkTitle = chunks.size() > 1
                ? title + " [" + (i + 1) + "/" + chunks.size() + "]"
                : title;
            ids.add(docDB.insert(chunkTitle, chunks.get(i), emb));
        }

        return ResponseEntity.ok(Map.of(
            "ids",    ids,
            "chunks", chunks.size(),
            "dims",   docDB.getDims()
        ));
    }

    // DELETE /doc/delete/{id}
    @DeleteMapping("/delete/{id}")
    public ResponseEntity<?> delete(@PathVariable int id) {
        return ResponseEntity.ok(Map.of("ok", docDB.delete(id)));
    }

    // GET /doc/list
    @GetMapping("/list")
    public ResponseEntity<?> list() {
        List<Map<String, Object>> result = docDB.getAll().stream()
            .map(d -> {
                String preview = d.text().length() > 120
                    ? d.text().substring(0, 120) + "…"
                    : d.text();
                int wordCount = d.text().trim().split("\\s+").length;
                return Map.<String, Object>of(
                    "id",      d.id(),
                    "title",   d.title(),
                    "preview", preview,
                    "words",   wordCount
                );
            })
            .collect(Collectors.toList());
        return ResponseEntity.ok(result);
    }

    // POST /doc/search  {"question":"...","k":3}
    @PostMapping("/search")
    public ResponseEntity<?> search(@RequestBody Map<String, Object> body) {
        String question = (String) body.getOrDefault("question", "");
        int    k        = ((Number) body.getOrDefault("k", 3)).intValue();

        if (question.isBlank()) return ResponseEntity.badRequest().body(Map.of("error", "question required"));

        float[] qEmb = ollama.embed(question);
        if (qEmb.length == 0) return ResponseEntity.status(503).body(Map.of("error", "Ollama unavailable"));

        List<DocumentDB.DocSearchResult> hits = docDB.search(qEmb, k, 0.7f);
        List<Map<String, Object>> contexts = hits.stream()
            .map(h -> Map.<String, Object>of(
                "id",       h.doc().id(),
                "title",    h.doc().title(),
                "distance", h.distance()
            ))
            .collect(Collectors.toList());

        return ResponseEntity.ok(Map.of("contexts", contexts));
    }

    // POST /doc/ask  {"question":"...","k":3}  — Full RAG pipeline
    @PostMapping("/ask")
    public ResponseEntity<?> ask(@RequestBody Map<String, Object> body) {
        String question = (String) body.getOrDefault("question", "");
        int    k        = ((Number) body.getOrDefault("k", 3)).intValue();

        if (question.isBlank()) return ResponseEntity.badRequest().body(Map.of("error", "question required"));

        // Step 1: embed question
        float[] qEmb = ollama.embed(question);
        if (qEmb.length == 0) return ResponseEntity.status(503).body(Map.of("error", "Ollama unavailable"));

        // Step 2: retrieve top-k chunks
        List<DocumentDB.DocSearchResult> hits = docDB.search(qEmb, k, 0.7f);

        // Step 3: build prompt with retrieved context
        StringBuilder ctx = new StringBuilder();
        for (int i = 0; i < hits.size(); i++) {
            ctx.append("[").append(i + 1).append("] ")
               .append(hits.get(i).doc().title()).append(":\n")
               .append(hits.get(i).doc().text()).append("\n\n");
        }

        String prompt =
            "You are a helpful assistant. Answer the user's question directly. " +
            "Use the provided context if it contains relevant information. " +
            "If it doesn't, just use your own general knowledge. " +
            "IMPORTANT: Do NOT mention the 'context' or say 'the context doesn't mention'. " +
            "Just answer the question naturally.\n\n" +
            "Context:\n" + ctx +
            "Question: " + question + "\n\nAnswer:";

        // Step 4: generate answer
        String answer = ollama.generate(prompt);

        // Step 5: return everything
        List<Map<String, Object>> contexts = hits.stream()
            .map(h -> Map.<String, Object>of(
                "id",       h.doc().id(),
                "title",    h.doc().title(),
                "text",     h.doc().text(),
                "distance", h.distance()
            ))
            .collect(Collectors.toList());

        return ResponseEntity.ok(Map.of(
            "answer",   answer,
            "model",    ollama.getGenModel(),
            "contexts", contexts,
            "docCount", docDB.size()
        ));
    }

    // GET /doc/status
    @GetMapping("/status")
    public ResponseEntity<?> status() {
        boolean up = ollama.isAvailable();
        return ResponseEntity.ok(Map.of(
            "ollamaAvailable", up,
            "embedModel",      ollama.getEmbedModel(),
            "genModel",        ollama.getGenModel(),
            "docCount",        docDB.size(),
            "docDims",         docDB.getDims()
        ));
    }
}
