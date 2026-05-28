
# This file contains code adapted from:
#
# Francisco N. F. Q. Simoes (2025)
# "The Minimal Search Space for Conditional Causal Bandits"
#https://github.com/francisco-simoes/minimal-set-conditional-intervention-bandits
#

import copy
from typing import Any, Set

import networkx as nx
from pgmpy.models import DiscreteBayesianNetwork


def C4(G: nx.DiGraph, U: Set) -> Set:
    """
    Computes L^\infty(U).
    :param G: A DAG.
    :param U: A set of nodes in G.
    :return: L^\infty(U).
    """

    connector = {node: None for node in G.nodes()}
    S = copy.copy(U)

    for v in U:
        connector[v] = v

    L = list(reversed(list(nx.topological_sort(G))))
    for v in L:
        if v in U:
            continue

        for c in G.successors(v):
            if connector[c] is not None:
                if connector[v] is None:
                    connector[v] = connector[c]
                elif connector[v] != connector[c]:
                    connector[v] = v
                    S.add(v)
                    break

    return S


def C4_on_target(bn: DiscreteBayesianNetwork, target: Any):
    """Apply C4 on (graph of bn, Pa(target))."""
    Pa_Y = bn.predecessors(target)
    print(Pa_Y)
    digraph: nx.DiGraph = bn.to_directed()
    mGISS = C4(digraph, set(Pa_Y))
    return mGISS