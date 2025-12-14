# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Testing
- `go test ./...` - Run all tests (note: currently some tests are failing)
- `go test ./internal/graph` - Run tests for the HNSW graph package
- `go test -v ./...` - Run tests with verbose output

### Building
- `go build ./...` - Build all packages
- `go mod tidy` - Clean up module dependencies
- `go fmt ./...` - Format all Go code

### Development
- `go run <file.go>` - Run a specific Go file
- `go vet ./...` - Check for common Go mistakes

## Architecture

This is an experimental implementation of the HNSW (Hierarchical Navigable Small Worlds) algorithm for approximate nearest neighbor search in Go.

### Core Components

**Package Structure:**
- `internal/graph/` - Core HNSW implementation
- `internal/graph/hnsw.go` - Main HNSW graph implementation
- `internal/graph/hnsw_test.go` - Test suite

**Key Types:**
- `Node` - Represents a point in the graph with ID, vector, and hierarchical friend connections
- `HNSW` - The main graph structure with thread-safe operations via mutex

**Core Operations:**
- `Insert(vector)` - Adds new vectors to the graph with level determination
- `Search(query, k)` - Finds k nearest neighbors using layer-wise search
- `EuclideanDistance` - Distance function for vector comparison

### Implementation Details

The HNSW uses a multi-layer graph structure where:
- Each node exists on multiple levels (determined probabilistically)
- Higher levels have fewer nodes but longer connections for fast traversal
- Level 0 contains all nodes and provides the final search results
- Thread-safe operations are ensured with mutex locks

The algorithm implements the standard HNSW approach with entry point search from top layer down to level 0, followed by efSearch expansion for finding nearest neighbors.

### Dependencies

- Uses `github.com/stretchr/testify` for testing assertions
- Standard Go libraries: `math`, `math/rand`, `sort`, `sync`