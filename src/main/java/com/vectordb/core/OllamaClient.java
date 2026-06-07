package com.vectordb.core;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.logging.Logger;

/**
 * OllamaClient — HTTP client wrapping Ollama's local REST API.
 *
 * Ollama exposes two endpoints we use:
 *   POST /api/embeddings  → converts text to a float[] vector
 *   POST /api/generate    → runs a local LLM to answer questions
 *
 * Uses Java's built-in HttpClient (java.net.http — introduced in Java 11).
 * No external HTTP library needed — this is a deliberate choice to reduce deps.
 *
 * Error handling philosophy:
 *   Return empty/null on failure rather than throwing.
 *   Callers check for empty and return a user-friendly error message.
 *   Ollama being offline should not crash the server.
 */
@Component
public class OllamaClient {

    private static final Logger log = Logger.getLogger(OllamaClient.class.getName());

    private static final String BASE_URL     = "http://127.0.0.1:11434";
    private static final String EMBED_MODEL  = "nomic-embed-text";
    private static final String GEN_MODEL    = "llama3.2";

    private final HttpClient   http;
    private final ObjectMapper mapper = new ObjectMapper();

    public OllamaClient() {
        this.http = HttpClient.newBuilder()
            .connectTimeout(Duration.ofSeconds(3))
            .build();
    }

    // ── Availability check ───────────────────────────────────────────────

    public boolean isAvailable() {
        try {
            HttpRequest req = HttpRequest.newBuilder()
                .uri(URI.create(BASE_URL + "/api/tags"))
                .timeout(Duration.ofSeconds(2))
                .GET()
                .build();
            HttpResponse<String> resp = http.send(req, HttpResponse.BodyHandlers.ofString());
            return resp.statusCode() == 200;
        } catch (Exception e) {
            return false;
        }
    }

    // ── Embedding ────────────────────────────────────────────────────────

    /**
     * Embed text into a float[] vector using nomic-embed-text.
     * Returns an empty array if Ollama is unavailable.
     *
     * nomic-embed-text produces 768-dimensional embeddings — the same
     * dimensionality as sentence-transformers models used in production.
     */
    public float[] embed(String text) {
        try {
            String body = mapper.writeValueAsString(
                new EmbedRequest(EMBED_MODEL, text));

            HttpRequest req = HttpRequest.newBuilder()
                .uri(URI.create(BASE_URL + "/api/embeddings"))
                .timeout(Duration.ofSeconds(30))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(body))
                .build();

            HttpResponse<String> resp = http.send(req, HttpResponse.BodyHandlers.ofString());
            if (resp.statusCode() != 200) return new float[0];

            JsonNode root = mapper.readTree(resp.body());
            JsonNode arr  = root.get("embedding");
            if (arr == null || !arr.isArray()) return new float[0];

            float[] embedding = new float[arr.size()];
            for (int i = 0; i < arr.size(); i++) embedding[i] = (float) arr.get(i).asDouble();
            return embedding;

        } catch (Exception e) {
            log.warning("Ollama embed failed: " + e.getMessage());
            return new float[0];
        }
    }

    // ── Generation ───────────────────────────────────────────────────────

    /**
     * Generate a text answer from the local LLM.
     * Returns an error string if Ollama is unavailable.
     */
    public String generate(String prompt) {
        try {
            String body = mapper.writeValueAsString(
                new GenerateRequest(GEN_MODEL, prompt, false));

            HttpRequest req = HttpRequest.newBuilder()
                .uri(URI.create(BASE_URL + "/api/generate"))
                .timeout(Duration.ofSeconds(180)) // LLMs are slow on CPU
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(body))
                .build();

            HttpResponse<String> resp = http.send(req, HttpResponse.BodyHandlers.ofString());
            if (resp.statusCode() != 200) return "ERROR: Ollama unavailable.";

            JsonNode root = mapper.readTree(resp.body());
            JsonNode r    = root.get("response");
            return r != null ? r.asText() : "ERROR: empty response";

        } catch (IOException | InterruptedException e) {
            log.warning("Ollama generate failed: " + e.getMessage());
            return "ERROR: Ollama unavailable. Run: ollama serve";
        }
    }

    public String getEmbedModel() { return EMBED_MODEL; }
    public String getGenModel()   { return GEN_MODEL;   }

    // ── Request records (Jackson serializes these automatically) ──────────
    private record EmbedRequest(String model, String prompt) {}
    private record GenerateRequest(String model, String prompt, boolean stream) {}
}
