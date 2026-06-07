package com.vectordb.model;

/**
 * Single result from a k-NN search.
 * Implements Comparable so results can be sorted by distance.
 */
public final class SearchResult implements Comparable<SearchResult> {

    private final int   id;
    private final float distance;
    private VectorItem  item; // populated by VectorDB after algorithm returns raw ids

    public SearchResult(int id, float distance) {
        this.id       = id;
        this.distance = distance;
    }

    public int        getId()       { return id; }
    public float      getDistance() { return distance; }
    public VectorItem getItem()     { return item; }
    public void       setItem(VectorItem item) { this.item = item; }

    @Override
    public int compareTo(SearchResult o) {
        return Float.compare(this.distance, o.distance);
    }
}
