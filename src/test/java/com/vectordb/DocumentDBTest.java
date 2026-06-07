package com.vectordb;

import com.vectordb.core.DocumentDB;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * DocumentDB Tests — focuses on the text chunking logic.
 *
 * Chunking is a critical RAG component. Wrong chunk sizes or broken
 * overlap logic silently degrades retrieval quality.
 * These tests pin the exact behaviour.
 */
class DocumentDBTest {

    @Test
    @DisplayName("Short text produces a single chunk (no splitting needed)")
    void testShortText() {
        String text = "Hello world this is a short document.";
        List<String> chunks = DocumentDB.chunkText(text, 250, 30);
        assertEquals(1, chunks.size());
        assertEquals(text.trim(), chunks.get(0));
    }

    @Test
    @DisplayName("Empty text returns empty list")
    void testEmptyText() {
        assertTrue(DocumentDB.chunkText("", 250, 30).isEmpty());
        assertTrue(DocumentDB.chunkText("   ", 250, 30).isEmpty());
    }

    @Test
    @DisplayName("Long text is split into overlapping chunks")
    void testLongTextChunking() {
        // Build a 300-word text
        StringBuilder sb = new StringBuilder();
        for (int i = 1; i <= 300; i++) sb.append("word").append(i).append(" ");
        String text = sb.toString().trim();

        List<String> chunks = DocumentDB.chunkText(text, 100, 20);

        // Should produce multiple chunks
        assertTrue(chunks.size() > 1, "300 words with chunk=100 must produce multiple chunks");
    }

    @Test
    @DisplayName("Overlapping chunks share words at boundaries")
    void testOverlapSharesWords() {
        // Build exact 200 words
        StringBuilder sb = new StringBuilder();
        for (int i = 1; i <= 200; i++) sb.append("word").append(i).append(" ");
        String text = sb.toString().trim();

        List<String> chunks = DocumentDB.chunkText(text, 100, 20);
        assertTrue(chunks.size() >= 2);

        // The last word of chunk[0] should appear in chunk[1]
        String[] chunk0Words = chunks.get(0).split("\\s+");
        String   lastWord    = chunk0Words[chunk0Words.length - 1];
        assertTrue(chunks.get(1).contains(lastWord),
            "Overlap: last word of chunk[0] must appear in chunk[1]");
    }

    @Test
    @DisplayName("Every word in original text appears in at least one chunk")
    void testNoWordsLost() {
        StringBuilder sb = new StringBuilder();
        for (int i = 1; i <= 500; i++) sb.append("word").append(i).append(" ");
        String text = sb.toString().trim();
        String[] originalWords = text.split("\\s+");

        List<String> chunks = DocumentDB.chunkText(text, 100, 20);
        String allChunks = String.join(" ", chunks);

        // Every word in the original must appear in at least one chunk
        for (String word : originalWords) {
            assertTrue(allChunks.contains(word),
                "Word '" + word + "' must appear in at least one chunk");
        }
    }
}
