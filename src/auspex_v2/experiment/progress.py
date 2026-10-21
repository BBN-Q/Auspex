"""Static planned point counts; completion is recorded only at POINT_END."""
from ._experiment_plan import CHILD_BLOCKS, walk_instructions


def point_total(document):
    def count(nodes):
        total = 0
        for node in nodes:
            op = node["op"]
            if op == "POINT_BEGIN":
                total += 1
            elif op == "SWEEP":
                inner = count(node["body"])
                if inner is None:
                    return None
                total += len(node["points"]) * inner
            elif any(child["op"] == "POINT_BEGIN" for block in CHILD_BLOCKS
                     for child in walk_instructions(node.get(block, []))):
                return None  # Conditional/dynamic point traversal: don't invent a denominator.
        return total
    return count(document["main"]) or None
