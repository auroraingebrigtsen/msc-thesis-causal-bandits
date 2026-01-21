from typing import Set, List, Tuple, FrozenSet, Dict, Optional
from cmab.utils.graphs.topological_sort import topological_sort


class CausalDiagram:
    """Causal Diagram represented as a DAG."""

    def __init__(
        self,
        nodes: Set[str],
        directed_edges: List[Tuple[str, str]],
        bidirected_edges: Optional[List[Tuple[str, str, str]]] = None,
    ):
        self.nodes = nodes
        self.directed_edges = directed_edges
        self.bidirected_edges = bidirected_edges or []
        self.parents: Dict[str, Set[str]] = {node: set() for node in nodes}
        self.children: Dict[str, Set[str]] = {node: set() for node in nodes}
        for u, v in directed_edges:
            self.parents[v].add(u)
            self.children[u].add(v)
        self.ancestors = self._compute_ancestors()
        self.descendants = self._compute_descendants()

        self.confounder_dict = {u: frozenset({x, y}) for x, y, u in self.bidirected_edges}
        self.bidirected_neighbors: Dict[str, Set[str]] = {n: set() for n in self.nodes}
        for x, y, _u in self.bidirected_edges:
            self.bidirected_neighbors[x].add(y)
            self.bidirected_neighbors[y].add(x)

    def _compute_ancestors(self) -> Dict[str, Set[str]]:
        """For each node v, return all nodes that have a directed path into v."""
        ancestors: Dict[str, Set[str]] = {v: set() for v in self.nodes}

        def dfs_up(start: str) -> Set[str]:
            seen: Set[str] = set()
            stack = list(self.parents[start])
            while stack:
                p = stack.pop()
                if p in seen:
                    continue
                seen.add(p)
                stack.extend(self.parents[p])
            return seen

        for v in self.nodes:
            ancestors[v] = dfs_up(v)

        return ancestors

    def _compute_descendants(self) -> Dict[str, Set[str]]:
        """For each node v, return all nodes reachable by a directed path out of v."""
        descendants: Dict[str, Set[str]] = {v: set() for v in self.nodes}

        def dfs_down(start: str) -> Set[str]:
            seen: Set[str] = set()
            stack = list(self.children[start])
            while stack:
                c = stack.pop()
                if c in seen:
                    continue
                seen.add(c)
                stack.extend(self.children[c])
            return seen

        for v in self.nodes:
            descendants[v] = dfs_down(v)

        return descendants

    def Pa(self, nodes: Set[str], include_self:bool=True) -> Set[str]:
        """Parents of a set of nodes."""
        parents = set().union(*(self.parents[n] for n in nodes))
        if include_self:
            return parents.union(nodes)
        return parents - nodes

    def An(self, Y: str, include_self:bool=True) -> Set[str]:
        """Ancestors of Y."""
        if include_self:
            return self.ancestors[Y].union({Y})
        return self.ancestors[Y]

    def  De(self, nodes: Set[str], include_self:bool=True) -> Set[str]:
        """Descendants of a set of nodes."""
        descendants = set().union(*(self.descendants[n] for n in nodes))
        if include_self:
            return descendants.union(nodes)
        return descendants
    

    def c_component(self, node: str) -> Set[str]:
        """Confounding component (bidirected-connected component) containing node."""
        if node not in self.nodes:
            raise KeyError(f"{node} not in graph")

        comp: Set[str] = set()
        stack = [node]

        while stack:
            v = stack.pop()
            if v in comp:
                continue
            comp.add(v)

            # add bidirected neighbors (observed nodes)
            for nbr in self.bidirected_neighbors.get(v, ()):
                if nbr not in comp:
                    stack.append(nbr)

        return comp
    
    def do(self, intervention_set: Set[str]) -> "CausalDiagram":
        """Return a new CausalDiagram after performing do(intervention_set)."""
        new_directed_edges = [
            (u, v) for (u, v) in self.directed_edges if v not in intervention_set
        ]
        new_bidirected_edges = [
            (x, y, u)
            for (x, y, u) in self.bidirected_edges
            if x not in intervention_set and y not in intervention_set
        ]
        return CausalDiagram(
            nodes=self.nodes,
            directed_edges=new_directed_edges,
            bidirected_edges=new_bidirected_edges,
        )

    def causal_order(self, backward=False) -> Tuple:
        top_to_bottom = topological_sort(self.nodes, self.directed_edges)
        if backward:
            return tuple(reversed(top_to_bottom))
        else:
            return tuple(top_to_bottom)
    
    def __getitem__(self, nodes: Set[str]) -> "CausalDiagram":
        """Subgraph induced by given nodes."""
        sub_directed_edges = [(u, v) for (u, v) in self.directed_edges if u in nodes and v in nodes]
        sub_bidirected_edges = [(x, y, u) for (x, y, u) in self.bidirected_edges if x in nodes and y in nodes]
        return CausalDiagram(nodes=set(nodes), directed_edges=sub_directed_edges, bidirected_edges=sub_bidirected_edges)
