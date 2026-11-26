package graph

import (
	"cmp"
	"container/heap"
	"math"
	"math/rand/v2"
	"slices"
	"sync"
)

// Node represents a point in the HNSW graph.
type Node struct {
	ID      int
	Vector  []float64
	Friends [][]*Node
}

// HNSW represents the HNSW graph.
type HNSW struct {
	Nodes     []*Node
	MaxEdges  int
	Ef        int
	LevelMult float64

	entryPoint *Node
	maxLevel   int
	mutex      sync.Mutex
}

// NewHNSW initializes a new HNSW graph.
func NewHNSW(maxEdges, ef int) *HNSW {
	return &HNSW{
		Nodes:     []*Node{},
		MaxEdges:  maxEdges,
		Ef:        ef,
		LevelMult: 1 / math.Log(float64(maxEdges)),
	}
}

// EuclideanDistance computes the Euclidean distance between two vectors.
func EuclideanDistance(vectorA, vectorB []float64) float64 {
	var sum float64
	for i := range vectorA {
		sum += (vectorA[i] - vectorB[i]) * (vectorA[i] - vectorB[i])
	}
	return math.Sqrt(sum)
}

// randomLevel draws a level from the exponential distribution used by HNSW.
func (h *HNSW) randomLevel() int {
	// rand.Float64() is in [0,1); guard against 0 which would yield +Inf.
	r := rand.Float64()
	if r <= 0 {
		r = math.SmallestNonzeroFloat64
	}
	return int(math.Floor(-math.Log(r) * h.LevelMult))
}

// Insert adds a new node to the HNSW graph.
func (h *HNSW) Insert(vector []float64) {
	h.mutex.Lock()
	defer h.mutex.Unlock()

	level := h.randomLevel()

	newNode := &Node{
		ID:      len(h.Nodes),
		Vector:  vector,
		Friends: make([][]*Node, level+1),
	}
	for i := range newNode.Friends {
		newNode.Friends[i] = []*Node{}
	}

	// First node becomes the entry point.
	if h.entryPoint == nil {
		h.Nodes = append(h.Nodes, newNode)
		h.entryPoint = newNode
		h.maxLevel = level
		return
	}

	entryPoint := h.entryPoint

	// Descend from the top of the graph down to just above the new node's
	// level, greedily moving to the closest node on each layer.
	for currentLevel := h.maxLevel; currentLevel > level; currentLevel-- {
		entryPoint = h.searchLayer(entryPoint, newNode.Vector, 1, currentLevel)[0]
	}

	// From the new node's level down to 0, find neighbors and connect.
	for currentLevel := min(level, h.maxLevel); currentLevel >= 0; currentLevel-- {
		neighbors := h.searchLayer(entryPoint, newNode.Vector, h.Ef, currentLevel)
		h.connectNewNode(newNode, neighbors, currentLevel)
		entryPoint = neighbors[0]
	}

	h.Nodes = append(h.Nodes, newNode)

	// Promote the new node to entry point if it reaches a higher level.
	if level > h.maxLevel {
		h.maxLevel = level
		h.entryPoint = newNode
	}
}

// Search finds the nearest neighbors to the query vector.
func (h *HNSW) Search(query []float64, k int) []*Node {
	h.mutex.Lock()
	defer h.mutex.Unlock()

	if h.entryPoint == nil {
		return nil
	}

	entryPoint := h.entryPoint
	for currentLevel := h.maxLevel; currentLevel >= 1; currentLevel-- {
		entryPoint = h.searchLayer(entryPoint, query, 1, currentLevel)[0]
	}

	ef := max(h.Ef, k)
	results := h.searchLayer(entryPoint, query, ef, 0)
	if len(results) > k {
		results = results[:k]
	}
	return results
}

// searchLayer performs a greedy best-first search on a single layer and
// returns up to ef nodes closest to the query, sorted nearest first.
func (h *HNSW) searchLayer(entryPoint *Node, query []float64, ef, layer int) []*Node {
	entryDist := EuclideanDistance(query, entryPoint.Vector)

	visited := map[int]struct{}{entryPoint.ID: {}}

	// candidates: min-heap ordered by distance (closest first).
	candidates := &minDistHeap{{node: entryPoint, dist: entryDist}}
	// results: max-heap ordered by distance (farthest first), size <= ef.
	results := &maxDistHeap{{node: entryPoint, dist: entryDist}}

	for candidates.Len() > 0 {
		closest := heap.Pop(candidates).(distNode)
		// Stop once the nearest candidate is farther than the farthest result.
		if closest.dist > (*results)[0].dist && results.Len() >= ef {
			break
		}

		// Skip if the candidate has no connections on this layer.
		if layer >= len(closest.node.Friends) {
			continue
		}

		for _, friend := range closest.node.Friends[layer] {
			if _, seen := visited[friend.ID]; seen {
				continue
			}
			visited[friend.ID] = struct{}{}

			dist := EuclideanDistance(query, friend.Vector)
			if results.Len() < ef || dist < (*results)[0].dist {
				heap.Push(candidates, distNode{node: friend, dist: dist})
				heap.Push(results, distNode{node: friend, dist: dist})
				if results.Len() > ef {
					heap.Pop(results)
				}
			}
		}
	}

	out := make([]*Node, results.Len())
	for i := len(out) - 1; i >= 0; i-- {
		out[i] = heap.Pop(results).(distNode).node
	}
	return out
}

// connectNewNode links newNode to its closest MaxEdges neighbors on a layer,
// keeping all edges bidirectional and pruning over-connected neighbors.
func (h *HNSW) connectNewNode(newNode *Node, candidates []*Node, level int) {
	if level >= len(newNode.Friends) {
		return
	}

	// candidates come sorted nearest-first; keep the closest MaxEdges.
	selected := candidates
	if len(selected) > h.MaxEdges {
		selected = selected[:h.MaxEdges]
	}

	for _, neighbor := range selected {
		if level >= len(neighbor.Friends) || neighbor == newNode {
			continue
		}

		newNode.Friends[level] = append(newNode.Friends[level], neighbor)
		neighbor.Friends[level] = append(neighbor.Friends[level], newNode)

		if len(neighbor.Friends[level]) > h.MaxEdges {
			h.pruneConnections(neighbor, level)
		}
	}
}

// pruneConnections trims a node's edges on a layer down to MaxEdges, keeping
// the closest neighbors and removing the dropped edges on both sides.
func (h *HNSW) pruneConnections(node *Node, level int) {
	friends := node.Friends[level]
	slices.SortFunc(friends, func(a, b *Node) int {
		return cmp.Compare(
			EuclideanDistance(node.Vector, a.Vector),
			EuclideanDistance(node.Vector, b.Vector),
		)
	})

	keep := friends[:h.MaxEdges]
	dropped := friends[h.MaxEdges:]
	node.Friends[level] = append([]*Node{}, keep...)

	// Remove the back-edges from the neighbors we dropped.
	for _, d := range dropped {
		if level >= len(d.Friends) {
			continue
		}
		edges := d.Friends[level]
		for i, f := range edges {
			if f == node {
				d.Friends[level] = append(edges[:i], edges[i+1:]...)
				break
			}
		}
	}
}

// distNode pairs a node with its distance to the query for the search heaps.
type distNode struct {
	node *Node
	dist float64
}

// minDistHeap is a min-heap of distNode ordered by distance (closest first).
type minDistHeap []distNode

func (h minDistHeap) Len() int           { return len(h) }
func (h minDistHeap) Less(i, j int) bool { return h[i].dist < h[j].dist }
func (h minDistHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *minDistHeap) Push(x any)        { *h = append(*h, x.(distNode)) }
func (h *minDistHeap) Pop() any {
	old := *h
	n := len(old)
	item := old[n-1]
	*h = old[:n-1]
	return item
}

// maxDistHeap is a max-heap of distNode ordered by distance (farthest first).
type maxDistHeap []distNode

func (h maxDistHeap) Len() int           { return len(h) }
func (h maxDistHeap) Less(i, j int) bool { return h[i].dist > h[j].dist }
func (h maxDistHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *maxDistHeap) Push(x any)        { *h = append(*h, x.(distNode)) }
func (h *maxDistHeap) Pop() any {
	old := *h
	n := len(old)
	item := old[n-1]
	*h = old[:n-1]
	return item
}
