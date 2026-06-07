package com.vectordb.model;

import java.util.Arrays;

/**
 * Immutable data model for a single vector entry.
 * Immutability is intentional — concurrent reads need no locking on the object itself.
 */
public final class VectorItem {

    private final int      id;
    private final String   metadata;
    private final String   category;
    private final float[]  embedding;

    public VectorItem(int id, String metadata, String category, float[] embedding) {
        this.id        = id;
        this.metadata  = metadata;
        this.category  = category;
        this.embedding = Arrays.copyOf(embedding, embedding.length); // defensive copy
    }

    public int     getId()        { return id; }
    public String  getMetadata()  { return metadata; }
    public String  getCategory()  { return category; }
    public float[] getEmbedding() { return Arrays.copyOf(embedding, embedding.length); }
    public int     getDims()      { return embedding.length; }

    @Override
    public String toString() {
        return "VectorItem{id=" + id + ", category='" + category + "', metadata='" + metadata + "'}";
    }
}
