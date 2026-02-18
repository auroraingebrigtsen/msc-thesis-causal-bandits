from cmab.utils.graphs.topological_sort import topological_sort
from collections import defaultdict
import networkx as nx

class CausalDiagram:
    """Causal Diagram represented as a DAG."""

    def __init__(
        self,
        nodes: set[str],
        directed_edges: list[tuple[str, str]],
        bidirected_edges: list[tuple[str, str, str]] = [],
        noise_vars: list[tuple[str, str]] = [],
    ):
        self.nodes = nodes
        self.directed_edges = directed_edges
        self.bidirected_edges = bidirected_edges
        self.noise_vars = noise_vars
        self.parents: dict[str, set[str]] = defaultdict(set)
        self.children: dict[str, set[str]] = defaultdict(set)
        for u, v in directed_edges:
            self.parents[v].add(u)
            self.children[u].add(v)
        self.ancestors = self._compute_ancestors()
        self.descendants = self._compute_descendants()

        self.confounder_dict = {u: frozenset({x, y}) for x, y, u in self.bidirected_edges}
        self.bidirected_neighbors = defaultdict(set)
        for x, y, _u in self.bidirected_edges:
            self.bidirected_neighbors[x].add(y)
            self.bidirected_neighbors[y].add(x)

    def _compute_ancestors(self) -> dict[str, set[str]]:
        """For each node v, return all nodes that have a directed path into v."""
        ancestors: dict[str, set[str]] = defaultdict(set)

        def dfs_up(start: str) -> set[str]:
            seen: set[str] = set()
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

    def _compute_descendants(self) -> dict[str, set[str]]:
        """For each node v, return all nodes reachable by a directed path out of v."""
        descendants: dict[str, set[str]] = defaultdict(set)

        def dfs_down(start: str) -> set[str]:
            seen: set[str] = set()
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

    def Pa(self, nodes: set[str], include_self:bool=True) -> set[str]:
        """Parents of a set of nodes."""
        parents = set().union(*(self.parents[n] for n in nodes))
        if include_self:
            return parents.union(nodes)
        return parents - nodes

    def An(self, Y: str, include_self:bool=True) -> set[str]:
        """Ancestors of Y."""
        if include_self:
            return self.ancestors[Y].union({Y})
        return self.ancestors[Y]

    def  De(self, nodes: set[str], include_self:bool=True) -> set[str]:
        """Descendants of a set of nodes."""
        descendants = set().union(*(self.descendants[n] for n in nodes))
        if include_self:
            return descendants.union(nodes)
        return descendants
    

    def c_component(self, node: str) -> set[str]:
        """Confounding component (bidirected-connected component) containing node."""
        if node not in self.nodes:
            raise KeyError(f"{node} not in graph")

        comp: set[str] = set()
        stack = [node]

        while stack:
            v = stack.pop()
            if v in comp:
                continue
            comp.add(v)

            # add bidirected neighbors (observed nodes)
            for nbr in self.bidirected_neighbors[v]:
                if nbr not in comp:
                    stack.append(nbr)

        return comp
    
    def do(self, intervention_set: set[str]) -> "CausalDiagram":
        """Return a new CausalDiagram after performing do(intervention_set)."""
        new_directed_edges = [
            (u, v) for (u, v) in self.directed_edges if v not in intervention_set
        ]
        new_bidirected_edges = [
            (x, y, u)
            for (x, y, u) in self.bidirected_edges
            if x not in intervention_set and y not in intervention_set
        ]

        new_noise_vars = []
        for (u,v) in self.noise_vars:
            if v in intervention_set:
                new_noise_vars.append((u, None))
            else:
                new_noise_vars.append((u, v))

        return CausalDiagram(
            nodes=self.nodes,
            directed_edges=new_directed_edges,
            bidirected_edges=new_bidirected_edges,
            noise_vars=new_noise_vars
        )

    def causal_order(self, backward=False) -> tuple:
        top_to_bottom = topological_sort(self.nodes, self.directed_edges)
        if backward:
            return tuple(reversed(top_to_bottom))
        else:
            return tuple(top_to_bottom)
        

    def d_separated(self, X: set[str], Y: set[str], Z: set[str]) -> bool:
        """Check if X and Y are d-separated given Z using networx. 
        Uses the augmented graph with exogenous noise variables"""
        G = nx.DiGraph()
        G.add_nodes_from(self.nodes)
        G.add_edges_from(self.directed_edges)

        for u, v in self.noise_vars:
            G.add_node(u)
            if v is not None:
                 G.add_edge(u, v)

        for x, y, u in self.bidirected_edges: 
            G.add_node(u)
            G.add_edge(u, x)
            G.add_edge(u, y)
        
        return nx.is_d_separator(G, X, Y, Z)

    def __getitem__(self, nodes: set[str]) -> "CausalDiagram":
        """Subgraph induced by given nodes."""
        sub_directed_edges = [(u, v) for (u, v) in self.directed_edges if u in nodes and v in nodes]
        sub_bidirected_edges = [(x, y, u) for (x, y, u) in self.bidirected_edges if x in nodes and y in nodes]
        sub_noise_vars = [(u, v) for (u, v) in self.noise_vars if u in nodes and (v in nodes or v is None)]
        return CausalDiagram(
            nodes=nodes,
            directed_edges=sub_directed_edges,
            bidirected_edges=sub_bidirected_edges,
            noise_vars=sub_noise_vars
        )