# HNSW

This is an experimental implementation of HNSW. Hierarchical Navigable Small Worlds (HNSW) algorithm is a graph-based approximate nearest neighbor search technique.

## Glossary

- Node struct: Represents a node with an ID, a vector, and a list of friends (neighbors) at each level.
- HNSW struct: Represents the HNSW graph, containing a list of nodes, parameters for the maximum number of edges per node (MaxEdges), the exploration factor (Ef), and the level multiplier for determining the level of nodes.
- NewHNSW: Initializes a new HNSW graph.
- EuclideanDistance: Calculates the Euclidean distance between two vectors.
- Insert: Adds a new node to the HNSW graph.
- Search: Finds the nearest neighbors to a given query vector.
- searchLayer: Helper function to search within a specific layer.
- connectNewNode: Connects a new node to its neighbors at a specific level.
