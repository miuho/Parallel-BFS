#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstddef>
#include <omp.h>

#include "CycleTimer.h"
#include "bfs.h"
#include "graph.h"

#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1
#define HYBRID_THRESHOLD_RATIO 0.03

void vertex_set_clear(vertex_set* list) {
    list->count = 0;
}

void vertex_set_init(vertex_set* list, int count) {
    list->alloc_count = count;
    list->present = (int*)malloc(sizeof(int) * list->alloc_count);
    vertex_set_clear(list);
}

/**
 * top-down step for hybrid bfs
 * algorithm same as hybrid, except that visited_array and changed boolean are
 * modified when new frontier is found
 */
void hybrid_top_down_step(
    vertex_set* frontier,
    vertex_set* new_frontier,
    bool *bu_frontier,
    int* changed,
    int* distances,
    int num_nodes,
    int num_edges,
    int* outgoing_starts,
    int* outgoing_edges)
{
    #pragma omp parallel for
    for (int i=0; i<frontier->count; i++) {

        int node = frontier->present[i];

        int start_edge = outgoing_starts[node];
        int end_edge = (node == num_nodes-1) ? num_edges : outgoing_starts[node+1];

        // attempt to add all neighbors to the new frontier
        for (int neighbor=start_edge; neighbor<end_edge; neighbor++) {
            int outgoing = outgoing_edges[neighbor];
            if (distances[outgoing] == NOT_VISITED_MARKER &&
                    __sync_bool_compare_and_swap(&distances[outgoing], 
                                                NOT_VISITED_MARKER,
                                                distances[node] + 1)
                                                == true) {
                *changed = 1;
                int index = __sync_add_and_fetch(&new_frontier->count, 1);
                new_frontier->present[index-1] = outgoing;
                bu_frontier[outgoing] = 1;
            }
        }
    }
}

/**
 * bottom-up step for hybrid bfs
 * algorithm same as hybrid, except that new_frontier array is updated each
 * step
 */
void hybrid_bottom_up_step(
    int* changed,
    int visitIter,
    bool *frontier,
    bool *new_frontier,
    int* distances,
    int num_nodes,
    int num_edges,
    int* incoming_starts,
    int* incoming_edges) {

    #pragma omp parallel for schedule(guided, 1000)
    for (int node=0; node<num_nodes; node++) {
        if (distances[node] != NOT_VISITED_MARKER) {
            continue;
        }

        int start_edge = incoming_starts[node];
        int end_edge = (node == num_nodes-1) ? num_edges : incoming_starts[node+1];

        // Check if any neighbor is visited before. If so, this vertex
        // becomes one of the new frontiers. Use the largest neighbor + 1
        // as distance
        for (int neighbor=start_edge; neighbor<end_edge; neighbor++) {
            int incoming = incoming_edges[neighbor];
            if (frontier[incoming]) {
                // Set this node's distance
                distances[node] = visitIter;
                new_frontier[node] = true;
                *changed = 1;
                break;
            }
        }

    }
}

void bfs_hybrid(graph* graph, solution* sol)
{
    /*******************Step 3********************/

    /////////////// top-down variables ///////////
    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set* frontier = &list1;
    vertex_set* new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
    for (int i=0; i<graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->present[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    bool *bu_frontier = (bool *)calloc(graph->num_nodes, sizeof(bool));
    bool *bu_new_frontier = (bool *)calloc(graph->num_nodes, sizeof(bool));
    
    /////////////// bottom-up variables ///////////
    // allocate visited array for bottom up
    int distance = 1;
    int changed = 1;
    int policy_changed = 0;

    // commonly used constants
    int num_nodes = graph->num_nodes;
    int num_edges = graph->num_edges;
    int *incoming_starts = graph->incoming_starts;
    int *incoming_edges = graph->incoming_edges;
    int *outgoing_starts = graph->outgoing_starts;
    int *outgoing_edges = graph->outgoing_edges;

    while (changed) {

#ifdef DEBUG
        double start_time = CycleTimer::currentSeconds();
#endif

        vertex_set_clear(new_frontier);
        changed = 0;

        // Use top-down if frontier is small, and use bottom-up if frontier
        // is large
        if (!policy_changed &&
            (static_cast<float>(frontier->count) /
                static_cast<float>(graph->num_nodes) < HYBRID_THRESHOLD_RATIO)) {
            hybrid_top_down_step(frontier, new_frontier, bu_frontier, 
                    &changed, sol->distances, num_nodes, num_edges,
                    outgoing_starts, outgoing_edges);
        } else {
            hybrid_bottom_up_step(&changed, distance, bu_frontier,
                    bu_new_frontier, sol->distances, num_nodes, num_edges,
                    incoming_starts, incoming_edges);
            policy_changed = 1;
        }

#ifdef DEBUG
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
        vertex_set* tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;

        if (policy_changed) {
            bool *tmp = bu_frontier;
            bu_frontier = bu_new_frontier;
            bu_new_frontier = tmp;
        }

        // increment visitIter
        distance++;
    }
}

// Take one step of "bottom_up" BFS.  For every vertex,
// follow all incoming edges, and add this vertex if neighboring
// vertex is in the frontier. 
int bottom_up_step(
    int distance,
    bool* frontier,
    bool* new_frontier,
    int* distances,
    int num_nodes,
    int num_edges,
    int* incoming_starts,
    int* incoming_edges) {

    int changed = 0;

    #pragma omp parallel for schedule(guided, 1000)
    for (int node=0; node<num_nodes; node++) {
        if (distances[node] != NOT_VISITED_MARKER) {
            continue;
        }

        int start_edge = incoming_starts[node];
        int end_edge = (node == num_nodes-1) ? num_edges : incoming_starts[node+1];

        // Check if any neighbor is visited before. If so, this vertex
        // becomes one of the new frontiers. 
        for (int neighbor=start_edge; neighbor<end_edge; neighbor++) {
            int incoming = incoming_edges[neighbor];
            if (frontier[incoming]) {
                // Set this node's distance
                distances[node] = distance;
                new_frontier[node] = true;
                changed = 1;
                break;
            }
        }
    }

    return changed;
}

void bfs_bottom_up(graph* graph, solution* sol)
{
    /*******************Step 2********************/
    bool *frontier = (bool *)calloc(graph->num_nodes, sizeof(bool));
    bool *new_frontier = (bool *)calloc(graph->num_nodes, sizeof(bool));

    // initialize all nodes to NOT_VISITED
    // initialize all frontiers and new_frontiers
    for (int i=0; i<graph->num_nodes; i++) {
        sol->distances[i] = NOT_VISITED_MARKER;
    }

    sol->distances[ROOT_NODE_ID] = 0;
    frontier[ROOT_NODE_ID] = true;

    // initialize
    int distance = 1;
    int changed = 1;

    // commonly used constants
    int num_nodes = graph->num_nodes;
    int num_edges = graph->num_edges;
    int *incoming_starts = graph->incoming_starts;
    int *incoming_edges = graph->incoming_edges;

    while (changed != 0) {

        changed = bottom_up_step(distance, frontier, new_frontier,
                sol->distances, num_nodes, num_edges, incoming_starts, 
                incoming_edges);

        bool *temp = frontier;
        frontier = new_frontier;
        new_frontier = temp;

        // increment iteration count
        distance++;
    }
}


// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
void top_down_step(
    vertex_set* frontier,
    vertex_set* new_frontier,
    int* distances,
    int num_nodes,
    int num_edges,
    int* outgoing_starts,
    int* outgoing_edges)

{
    #pragma omp parallel for
    for (int i=0; i<frontier->count; i++) {

        int node = frontier->present[i];

        int start_edge = outgoing_starts[node];
        int end_edge = (node == num_nodes-1) ? num_edges : outgoing_starts[node+1];

        // attempt to add all neighbors to the new frontier
        for (int neighbor=start_edge; neighbor<end_edge; neighbor++) {
            int outgoing = outgoing_edges[neighbor];
            if ((distances[outgoing] == NOT_VISITED_MARKER) &&
                    __sync_bool_compare_and_swap(&distances[outgoing], 
                                                NOT_VISITED_MARKER,
                                                distances[node] + 1) 
                                                == true) {
                int index = __sync_add_and_fetch(&new_frontier->count, 1);
                new_frontier->present[index-1] = outgoing;
            }
        }
    }
}

// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(graph* graph, solution* sol) {
    
    /*******************Step 1********************/

    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set* frontier = &list1;
    vertex_set* new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
    for (int i=0; i<graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->present[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    // commonly used constants
    int num_nodes = graph->num_nodes;
    int num_edges = graph->num_edges;
    int *outgoing_starts = graph->outgoing_starts;
    int *outgoing_edges = graph->outgoing_edges;

    while (frontier->count != 0) {

#ifdef DEBUG
        double start_time = CycleTimer::currentSeconds();
#endif

        vertex_set_clear(new_frontier);

        top_down_step(frontier, new_frontier, sol->distances, num_nodes,
                num_edges, outgoing_starts, outgoing_edges);

#ifdef DEBUG
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
        vertex_set* tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }
}
