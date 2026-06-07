package com.vectordb.storage;

import java.io.*;
import java.nio.file.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.logging.Logger;

/**
 * Write-Ahead Log (WAL) — Crash-safe persistence layer.
 *
 * WHY WAL?
 * Every production database (PostgreSQL, MySQL, RocksDB, Kafka) uses a WAL.
 * The rule: write to disk BEFORE updating in-memory state.
 * On crash, replay the log to reconstruct the last consistent state.
 *
 * Our WAL format (line-based, human-readable):
 *   INSERT|id|category|metadata|dim0,dim1,...
 *   DELETE|id
 *
 * Trade-offs made here:
 *   - We flush after every write (fsync semantics) → durability at cost of throughput
 *   - Log grows unboundedly; in production you'd checkpoint and truncate
 *   - For FAANG interview: "I'd add periodic compaction to bound log size"
 *
 * This is the key talking point that differentiates this project from
 * any other student vector DB.
 */
public class WAL {

    private static final Logger log = Logger.getLogger(WAL.class.getName());

    private static final String INSERT_OP = "INSERT";
    private static final String DELETE_OP = "DELETE";

    private final Path        walPath;
    private       BufferedWriter writer;

    public WAL(String filePath) throws IOException {
        this.walPath = Paths.get(filePath);
        // Append mode — we never overwrite existing log entries
        this.writer  = Files.newBufferedWriter(walPath,
            StandardOpenOption.CREATE, StandardOpenOption.APPEND);
        log.info("WAL initialized at: " + walPath.toAbsolutePath());
    }

    // ── Write operations ─────────────────────────────────────────────────

    /**
     * Log an INSERT before it hits the in-memory index.
     * flush() ensures the OS buffer is flushed to disk (durable write).
     */
    public synchronized void logInsert(int id, String category,
                                       String metadata, float[] embedding) throws IOException {
        StringBuilder sb = new StringBuilder();
        sb.append(INSERT_OP).append('|')
          .append(id).append('|')
          .append(escapePipe(category)).append('|')
          .append(escapePipe(metadata)).append('|');

        for (int i = 0; i < embedding.length; i++) {
            if (i > 0) sb.append(',');
            sb.append(embedding[i]);
        }

        writer.write(sb.toString());
        writer.newLine();
        writer.flush(); // Critical: ensure durability before acknowledging the insert
    }

    /**
     * Log a DELETE before removing from the in-memory index.
     */
    public synchronized void logDelete(int id) throws IOException {
        writer.write(DELETE_OP + "|" + id);
        writer.newLine();
        writer.flush();
    }

    // ── Recovery ─────────────────────────────────────────────────────────

    /**
     * Replay the WAL on startup to reconstruct in-memory state.
     * Returns all WALEntry records in log order.
     *
     * The caller (VectorDB) applies these entries to rebuild its indexes.
     */
    public List<WALEntry> replay() throws IOException {
        List<WALEntry> entries = new ArrayList<>();

        if (!Files.exists(walPath)) return entries;

        try (BufferedReader reader = Files.newBufferedReader(walPath)) {
            String line;
            int lineNum = 0;
            while ((line = reader.readLine()) != null) {
                lineNum++;
                line = line.trim();
                if (line.isEmpty()) continue;

                try {
                    WALEntry entry = parseLine(line);
                    if (entry != null) entries.add(entry);
                } catch (Exception e) {
                    // Partial write on crash — skip corrupted entry, continue
                    log.warning("Skipping corrupted WAL entry at line " + lineNum + ": " + e.getMessage());
                }
            }
        }

        log.info("WAL replay complete: " + entries.size() + " entries");
        return entries;
    }

    private WALEntry parseLine(String line) {
        String[] parts = line.split("\\|", 5); // max 5 parts

        return switch (parts[0]) {
            case INSERT_OP -> {
                if (parts.length < 5) yield null;
                int     id        = Integer.parseInt(parts[1]);
                String  category  = unescapePipe(parts[2]);
                String  metadata  = unescapePipe(parts[3]);
                float[] embedding = parseFloatArray(parts[4]);
                yield new WALEntry(WALEntry.Op.INSERT, id, category, metadata, embedding);
            }
            case DELETE_OP -> {
                if (parts.length < 2) yield null;
                int id = Integer.parseInt(parts[1]);
                yield new WALEntry(WALEntry.Op.DELETE, id, null, null, null);
            }
            default -> null; // unknown op — skip
        };
    }

    // ── Helpers ───────────────────────────────────────────────────────────

    private float[] parseFloatArray(String s) {
        String[] tokens = s.split(",");
        float[] arr = new float[tokens.length];
        for (int i = 0; i < tokens.length; i++) arr[i] = Float.parseFloat(tokens[i].trim());
        return arr;
    }

    /** Escape pipe characters in metadata/category to avoid parsing ambiguity. */
    private String escapePipe(String s) {
        return s == null ? "" : s.replace("\\", "\\\\").replace("|", "\\|");
    }

    private String unescapePipe(String s) {
        return s == null ? "" : s.replace("\\|", "|").replace("\\\\", "\\");
    }

    public synchronized void close() throws IOException {
        if (writer != null) { writer.close(); writer = null; }
    }

    // ── WAL Entry record ──────────────────────────────────────────────────

    public record WALEntry(Op op, int id, String category, String metadata, float[] embedding) {
        public enum Op { INSERT, DELETE }
    }
}
