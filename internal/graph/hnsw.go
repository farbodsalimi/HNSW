package graph

import (
	"math"
	"math/rand"
	"sort"
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
	mutex     sync.Mutex
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

// Insert adds a new node to the HNSW graph.
func (h *HNSW) Insert(vector []float64) {
	h.mutex.Lock()
	defer h.mutex.Unlock()

	newNode := &Node{
		ID:      len(h.Nodes),
		Vector:  vector,
		Friends: [][]*Node{},
	}

	// Determine the level of the new node
	level := int(math.Floor(-math.Log(rand.Float64()) * h.LevelMult))
	for i := 0; i <= level; i++ {
		newNode.Friends = append(newNode.Friends, []*Node{})
	}

	if len(h.Nodes) == 0 {
		h.Nodes = append(h.Nodes, newNode)
		return
	}

	entryPoint := h.Nodes[0]
	for currentLevel := len(entryPoint.Friends) - 1; currentLevel > level; currentLevel-- {
		entryPoint = h.searchLayer(entryPoint, newNode.Vector, 1, currentLevel)[0]
	}

	for currentLevel := level; currentLevel >= 0; currentLevel-- {
		neighbors := h.searchLayer(entryPoint, newNode.Vector, h.Ef, currentLevel)
		h.connectNewNode(newNode, neighbors, currentLevel)
	}

	h.Nodes = append(h.Nodes, newNode)
}

// Search finds the nearest neighbors to the query vector.
func (h *HNSW) Search(query []float64, k int) []*Node {
	h.mutex.Lock()
	defer h.mutex.Unlock()

	entryPoint := h.Nodes[0]
	for currentLevel := len(entryPoint.Friends) - 1; currentLevel >= 1; currentLevel-- {
		entryPoint = h.searchLayer(entryPoint, query, 1, currentLevel)[0]
	}

	return h.searchLayer(entryPoint, query, k, 0)
}

func (h *HNSW) searchLayer(entryPoint *Node, query []float64, ef, layer int) []*Node {
	candidates := []*Node{entryPoint}
	visited := make(map[int]struct{})
	visited[entryPoint.ID] = struct{}{}

	results := make([]*Node, 0, ef)
	results = append(results, entryPoint)
	sort.Slice(results, func(i, j int) bool {
		return EuclideanDistance(query, results[i].Vector) < EuclideanDistance(query, results[j].Vector)
	})

	for len(candidates) > 0 {
		candidate := candidates[0]
		candidates = candidates[1:]

		for _, friend := range candidate.Friends[layer] {
			if _, seen := visited[friend.ID]; seen {
				continue
			}
			visited[friend.ID] = struct{}{}

			if len(results) < ef {
				results = append(results, friend)
				candidates = append(candidates, friend)
			} else if EuclideanDistance(query, friend.Vector) < EuclideanDistance(query, results[len(results)-1].Vector) {
				results[len(results)-1] = friend
				candidates = append(candidates, friend)
				sort.Slice(results, func(i, j int) bool {
					return EuclideanDistance(query, results[i].Vector) < EuclideanDistance(query, results[j].Vector)
				})
			}
		}
	}

	if len(results) > ef {
		results = results[:ef]
	}

	return results
}

func (h *HNSW) connectNewNode(newNode *Node, neighbors []*Node, level int) {
	for _, neighbor := range neighbors {
		if len(neighbor.Friends[level]) < h.MaxEdges {
			neighbor.Friends[level] = append(neighbor.Friends[level], newNode)
		} else {
			maxDistance := 0.0
			maxIndex := 0
			for i, friend := range neighbor.Friends[level] {
				distance := EuclideanDistance(neighbor.Vector, friend.Vector)
				if distance > maxDistance {
					maxDistance = distance
					maxIndex = i
				}
			}

			if EuclideanDistance(neighbor.Vector, newNode.Vector) < maxDistance {
				neighbor.Friends[level][maxIndex] = newNode
			}
		}
		newNode.Friends[level] = append(newNode.Friends[level], neighbor)
	}
}
