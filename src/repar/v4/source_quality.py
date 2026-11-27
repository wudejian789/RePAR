"""Map externally supplied unavailable text ranges to the existing source leaves."""
from bisect import bisect_right

from .memory import merge_ranges


def unavailable_leaves(index, ranges):
    if not ranges:
        return frozenset()
    if any(not 0 <= start < end <= len(index.context) for start, end in ranges):
        raise ValueError("Unavailable source range is outside the original context")
    clipped = merge_ranges(ranges)
    ends = [end for _, end in clipped]
    unavailable = set()
    for i, (start, end) in enumerate(index.spans):
        size = 0
        for lo, hi in clipped[bisect_right(ends, start):]:
            if lo >= end:
                break
            size += min(hi, end) - max(lo, start)
        if size * 2 > end - start:
            unavailable.add(f"C{i}")
    return frozenset(unavailable)
