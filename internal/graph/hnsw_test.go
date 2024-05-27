package graph_test

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/farbodsalimi/HNSW/internal/graph"
)

// Test insertion of a single node and ensure it exists in the graph.
func TestInsertSingleNode(t *testing.T) {
	hnsw := graph.NewHNSW(5, 200)
	vector := []float64{1, 2}
	hnsw.Insert(vector)

	assert.Equal(t, 1, len(hnsw.Nodes), "Expected 1 node")
	assert.Equal(t, vector, hnsw.Nodes[0].Vector, "Expected node vector to be the same as inserted vector")
}

// Test insertion of multiple nodes and ensure they all exist in the graph.
func TestInsertMultipleNodes(t *testing.T) {
	hnsw := graph.NewHNSW(5, 200)
	vectors := [][]float64{
		{1, 2}, {3, 4}, {5, 6},
	}

	for _, vector := range vectors {
		hnsw.Insert(vector)
	}

	assert.Equal(t, len(vectors), len(hnsw.Nodes), "Expected number of nodes to be equal to the number of inserted vectors")

	for i, vector := range vectors {
		assert.Equal(t, vector, hnsw.Nodes[i].Vector, "Expected node vector to be the same as inserted vector")
	}
}

// Test the search function to find the nearest neighbor of a query point.
func TestSearchNearestNeighbor(t *testing.T) {
	hnsw := graph.NewHNSW(5, 200)
	vectors := [][]float64{
		{1, 2}, {2, 3}, {3, 4},
	}
	for _, vector := range vectors {
		hnsw.Insert(vector)
	}

	query := []float64{2, 2.5}
	neighbors := hnsw.Search(query, 1)

	assert.Equal(t, 1, len(neighbors), "Expected 1 neighbor")

	expectedNeighbor := vectors[1]
	assert.Equal(t, expectedNeighbor, neighbors[0].Vector, "Expected nearest neighbor to be the closest vector")
}

// Test the search function to find multiple nearest neighbors of a query point.
func TestSearchMultipleNearestNeighbors(t *testing.T) {
	hnsw := graph.NewHNSW(5, 200)
	vectors := [][]float64{
		{1, 2}, {2, 3}, {3, 4},
	}
	for _, vector := range vectors {
		hnsw.Insert(vector)
	}

	query := []float64{2, 2.5}
	neighbors := hnsw.Search(query, 2)

	assert.Equal(t, 2, len(neighbors), "Expected 2 neighbors")

	expectedNeighbors := [][]float64{
		{2, 3}, {1, 2},
	}
	for i, expectedVector := range expectedNeighbors {
		assert.Equal(t, expectedVector, neighbors[i].Vector, "Expected neighbor to be the closest vector")
	}
}

// Test that the HNSW graph maintains correct connections at different levels.
func TestGraphConnections(t *testing.T) {
	hnsw := graph.NewHNSW(5, 200)
	vectors := [][]float64{
		{1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6},
	}
	for _, vector := range vectors {
		hnsw.Insert(vector)
	}

	for _, node := range hnsw.Nodes {
		for level, friends := range node.Friends {
			for _, friend := range friends {
				found := false
				for _, friendFriend := range friend.Friends[level] {
					if friendFriend == node {
						found = true
						break
					}
				}
				assert.True(t, found, "Node %d is not correctly connected to its friend %d at level %d", node.ID, friend.ID, level)
			}
		}
	}
}
