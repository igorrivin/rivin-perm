"""Independent transfer check for the periodic-x/y 4x4x5 dimer count.

This verifier deliberately does not import or call the permanent code.  It
processes the five 4x4 toroidal layers along the open z direction.  For each
incoming vertical-edge mask, a recursive layer transition either pairs the
first uncovered vertex horizontally or sends it to the next layer.

The expected runtime is a few seconds on the GH200 host CPU.
"""

from functools import lru_cache
import json
import time


WIDTH = 4
HEIGHT = 4
LAYERS = 5
VERTICES = WIDTH * HEIGHT
FULL_MASK = (1 << VERTICES) - 1
EXPECTED = 2_551_276_235_535_120


def _toroidal_neighbors():
    neighbors = []
    for y in range(HEIGHT):
        for x in range(WIDTH):
            adjacent = {
                y * WIDTH + (x - 1) % WIDTH,
                y * WIDTH + (x + 1) % WIDTH,
                ((y - 1) % HEIGHT) * WIDTH + x,
                ((y + 1) % HEIGHT) * WIDTH + x,
            }
            neighbors.append(tuple(sorted(adjacent)))
    return tuple(neighbors)


NEIGHBORS = _toroidal_neighbors()


@lru_cache(maxsize=None)
def _layer_transitions(incoming):
    """Map an incoming vertical mask to outgoing-mask multiplicities."""
    counts = {}

    def visit(covered, outgoing):
        if covered == FULL_MASK:
            counts[outgoing] = counts.get(outgoing, 0) + 1
            return

        free = (~covered) & FULL_MASK
        vertex_bit = free & -free
        vertex = vertex_bit.bit_length() - 1

        # Send this vertex through a vertical dimer to the next layer.
        visit(covered | vertex_bit, outgoing | vertex_bit)

        # Or pair it along one of the four in-layer torus edges.
        for neighbor in NEIGHBORS[vertex]:
            neighbor_bit = 1 << neighbor
            if not covered & neighbor_bit:
                visit(covered | vertex_bit | neighbor_bit, outgoing)

    visit(incoming, 0)
    return counts


@lru_cache(maxsize=None)
def _last_layer_count(incoming):
    """Count completions with no outgoing vertical edge."""
    def visit(covered):
        if covered == FULL_MASK:
            return 1
        free = (~covered) & FULL_MASK
        vertex_bit = free & -free
        vertex = vertex_bit.bit_length() - 1
        total = 0
        for neighbor in NEIGHBORS[vertex]:
            neighbor_bit = 1 << neighbor
            if not covered & neighbor_bit:
                total += visit(covered | vertex_bit | neighbor_bit)
        return total

    return visit(incoming)


def transfer_count():
    """Return the exact dimer count and state counts after each transition."""
    states = {0: 1}
    state_counts = []
    for _layer in range(LAYERS - 1):
        next_states = {}
        for incoming, prefix_count in states.items():
            for outgoing, multiplicity in _layer_transitions(incoming).items():
                next_states[outgoing] = (
                    next_states.get(outgoing, 0)
                    + prefix_count * multiplicity
                )
        states = next_states
        state_counts.append(len(states))

    result = sum(
        prefix_count * _last_layer_count(incoming)
        for incoming, prefix_count in states.items()
    )
    return result, tuple(state_counts)


def main():
    started = time.perf_counter()
    result, state_counts = transfer_count()
    elapsed = time.perf_counter() - started
    if result != EXPECTED:
        raise RuntimeError(
            f"transfer result {result} does not match expected {EXPECTED}"
        )
    print(json.dumps({
        "result": result,
        "seconds": elapsed,
        "states_after_transitions": state_counts,
        "cached_transition_states": _layer_transitions.cache_info().currsize,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
