package com.vectordb.api;

import com.vectordb.core.DocumentDB;
import com.vectordb.core.OllamaClient;
import com.vectordb.core.VectorDB;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.LinkedHashMap;
import java.util.Map;

@RestController
@CrossOrigin(origins = "*")
@RequestMapping("/api/status")
public class StatusController {

    private final VectorDB db;
    private final DocumentDB docDB;
    private final OllamaClient ollama;

    public StatusController(VectorDB db, DocumentDB docDB, OllamaClient ollama) {
        this.db = db;
        this.docDB = docDB;
        this.ollama = ollama;
    }

    @GetMapping
    public ResponseEntity<?> status() {

        Map<String, Object> resp = new LinkedHashMap<>();

        try {
            resp.put("ollamaAvailable", ollama.isAvailable());
        } catch (Exception e) {
            resp.put("ollamaAvailable", false);
            resp.put("ollamaError", e.getMessage());
        }

        try {
            resp.put("embedModel", ollama.getEmbedModel());
        } catch (Exception e) {
            resp.put("embedModel", "ERROR");
        }

        try {
            resp.put("genModel", ollama.getGenModel());
        } catch (Exception e) {
            resp.put("genModel", "ERROR");
        }

        resp.put("docCount", docDB.size());
        resp.put("docDims", docDB.getDims());
        resp.put("demoDims", db.getDims());
        resp.put("demoCount", db.size());
        resp.put("walEnabled", true);

        return ResponseEntity.ok(resp);
    }
}