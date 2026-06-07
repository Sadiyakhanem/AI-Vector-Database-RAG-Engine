package com.vectordb;

import com.vectordb.storage.WAL;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * WAL (Write-Ahead Log) Tests
 *
 * These tests simulate crash recovery — the most important
 * correctness property of a persistent database.
 *
 * The key invariant: after writing to the WAL and "crashing"
 * (closing and reopening), all entries must be recoverable.
 *
 * This is how PostgreSQL, RocksDB, and Kafka test their WALs.
 */
class WALTest {

    private static final String WAL_PATH = "test_vectordb.wal";
    private WAL wal;

    @BeforeEach
    void setUp() throws IOException {
        Files.deleteIfExists(Paths.get(WAL_PATH));
        wal = new WAL(WAL_PATH);
    }

    @AfterEach
    void tearDown() throws IOException {
        wal.close();
        Files.deleteIfExists(Paths.get(WAL_PATH));
    }

    @Test
    @DisplayName("INSERT entry survives WAL replay (crash recovery)")
    void testInsertSurvivesReplay() throws IOException {
        float[] emb = {0.1f, 0.2f, 0.3f, 0.4f};
        wal.logInsert(42, "cs", "Binary Search Tree", emb);

        // Simulate crash: close and reopen
        wal.close();
        WAL recovered = new WAL(WAL_PATH);
        List<WAL.WALEntry> entries = recovered.replay();
        recovered.close();

        assertEquals(1, entries.size());
        WAL.WALEntry e = entries.get(0);
        assertEquals(WAL.WALEntry.Op.INSERT, e.op());
        assertEquals(42,              e.id());
        assertEquals("cs",            e.category());
        assertEquals("Binary Search Tree", e.metadata());
        assertArrayEquals(emb, e.embedding(), 1e-5f);
    }

    @Test
    @DisplayName("DELETE entry survives WAL replay")
    void testDeleteSurvivesReplay() throws IOException {
        wal.logInsert(1, "cat", "item", new float[]{0.5f, 0.5f});
        wal.logDelete(1);

        wal.close();
        WAL recovered = new WAL(WAL_PATH);
        List<WAL.WALEntry> entries = recovered.replay();
        recovered.close();

        assertEquals(2, entries.size());
        assertEquals(WAL.WALEntry.Op.INSERT, entries.get(0).op());
        assertEquals(WAL.WALEntry.Op.DELETE, entries.get(1).op());
        assertEquals(1, entries.get(1).id());
    }

    @Test
    @DisplayName("Multiple entries preserved in order")
    void testMultipleEntriesOrdered() throws IOException {
        for (int i = 1; i <= 10; i++) {
            wal.logInsert(i, "cat" + i, "meta" + i, new float[]{(float)i, (float)i});
        }
        wal.logDelete(5);

        wal.close();
        WAL recovered = new WAL(WAL_PATH);
        List<WAL.WALEntry> entries = recovered.replay();
        recovered.close();

        assertEquals(11, entries.size()); // 10 inserts + 1 delete

        // Verify order is preserved
        for (int i = 0; i < 10; i++) {
            assertEquals(WAL.WALEntry.Op.INSERT, entries.get(i).op());
            assertEquals(i + 1, entries.get(i).id());
        }
        assertEquals(WAL.WALEntry.Op.DELETE, entries.get(10).op());
        assertEquals(5, entries.get(10).id());
    }

    @Test
    @DisplayName("Metadata with special chars (pipes, backslashes) survives round-trip")
    void testSpecialCharactersInMetadata() throws IOException {
        String tricky = "Item with | pipe and \\ backslash and \"quotes\"";
        wal.logInsert(1, "cat", tricky, new float[]{1f, 0f});

        wal.close();
        WAL recovered = new WAL(WAL_PATH);
        List<WAL.WALEntry> entries = recovered.replay();
        recovered.close();

        assertEquals(1, entries.size());
        assertEquals(tricky, entries.get(0).metadata());
    }

    @Test
    @DisplayName("Empty WAL replay returns empty list")
    void testEmptyWALReplay() throws IOException {
        List<WAL.WALEntry> entries = wal.replay();
        assertTrue(entries.isEmpty());
    }

    @Test
    @DisplayName("High-dimensional embedding (768D) survives round-trip")
    void testHighDimensionalEmbedding() throws IOException {
        float[] emb768 = new float[768];
        for (int i = 0; i < 768; i++) emb768[i] = (float) Math.sin(i * 0.01);

        wal.logInsert(1, "doc", "Ollama nomic-embed-text 768D vector", emb768);
        wal.close();

        WAL recovered = new WAL(WAL_PATH);
        List<WAL.WALEntry> entries = recovered.replay();
        recovered.close();

        assertEquals(1, entries.size());
        assertArrayEquals(emb768, entries.get(0).embedding(), 1e-5f,
            "768D embedding must survive WAL round-trip without precision loss");
    }

    @Test
    @DisplayName("WAL file is created on disk")
    void testWALFileExists() throws IOException {
        wal.logInsert(1, "cat", "meta", new float[]{1f});
        assertTrue(Files.exists(Paths.get(WAL_PATH)), "WAL file must exist on disk after write");
        assertTrue(Files.size(Paths.get(WAL_PATH)) > 0, "WAL file must be non-empty");
    }
}
