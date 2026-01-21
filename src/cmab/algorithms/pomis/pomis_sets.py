# This file contains code adapted from:
#
# Sanghack Lee (2018)
# "Structural Causal Bandits: Where to Intervene?"
# https://github.com/sanghack81/SCMMAB-NIPS2018
#
# Licensed under the MIT License.
# See the LICENSE file in the repository root for the full license text.
#
# Modifications:
# - 
# - 

from cmab.scm.causal_diagram import CausalDiagram
from typing import FrozenSet, Set, List, Tuple, AbstractSet, Any
from itertools import combinations


### Helpers ### 

def pop(xs: Set):
    x = next(iter(xs))
    xs.remove(x)
    return x

def only(W: list[Any], Z: set[Any]) -> list[Any]:
    if not Z:
        return []
    return [w for w in W if w in Z]



### Functions to compute different POMIS sets ###

def MISs(G: CausalDiagram, Y: str) -> FrozenSet[FrozenSet[str]]:
    """ All minimal intervention sets """
    II = G.V - {Y}
    assert II <= G.V
    assert Y not in II

    G = G[G.An(Y)]
    Ws = G.causal_order(backward=True)
    Ws = only(Ws, II)
    return subMISs(G, Y, frozenset(), Ws)


def subMISs(G: CausalDiagram, Y: str, Xs: FrozenSet[str], Ws: List[str]) -> FrozenSet[FrozenSet[str]]:
    """ subroutine for MISs -- this creates a recursive call tree with n, n-1, n-2, ... widths """
    out = frozenset({Xs})
    for i, W_i in enumerate(Ws):
        H = G.do({W_i})
        H = H[H.An(Y)]
        out |= subMISs(H, Y, Xs | {W_i}, only(Ws[i + 1:], H.V))
    return out

def MUCT(G: CausalDiagram, Y: str) -> FrozenSet[str]:
    """ Minimal Unobserved Confounder's Territory """
    H = G[G.An(Y)]

    Qs = {Y}
    Ts = frozenset({Y})
    while Qs:
        q = pop(Qs)
        W =H.c_component(q)
        Ts |= W
        Qs = (Qs | H.De(W, include_self=False)) - Ts

    return Ts

def IB(G: CausalDiagram, Y: str) -> FrozenSet[str]:
    Zs = MUCT(G, Y)
    return G.Pa(Zs) - Zs

def MUCT_IB(G: CausalDiagram, Y) -> Tuple[FrozenSet[str], FrozenSet[str]]:
    Zs = MUCT(G, Y)
    return Zs, G.Pa(Zs) - Zs
    
def subPOMISs(G: CausalDiagram, Y, Ws: List, obs=None) -> Set[FrozenSet[str]]:
    if obs is None:
        obs = set()

    out = []
    for i, W_i in enumerate(Ws):
        Ts, Xs = MUCT_IB(G.do({W_i}), Y)
        new_obs = obs | set(Ws[:i])
        if not (Xs & new_obs):
            out.append(Xs)
            new_Ws = only(Ws[i + 1:], Ts)
            if new_Ws:
                out.extend(subPOMISs(G.do(Xs)[Ts | Xs], Y, new_Ws, new_obs))
    return {frozenset(_) for _ in out}


def POMISs(G: CausalDiagram, Y: str) -> List[FrozenSet[str]]:
        """ Precompute all POMISs for the given causal diagram G and target variable Y """
        G = G[G.An(Y)]

        Ts, Xs = MUCT_IB(G, Y)
        H = G.do(Xs)[Ts | Xs]
        return subPOMISs(H, Y, only(H.causal_order(backward=True), Ts - {Y})) | {frozenset(Xs)}

def bruteforce_POMISs(G: CausalDiagram, Y: str) -> FrozenSet[FrozenSet[str]]:
    """ This computes a complete set of POMISs in a brute-force way """
    return frozenset({frozenset(IB(G.do(Ws), Y))
                      for Ws in combinations(list(G.V - {Y}))})


def minimal_do(G: CausalDiagram, Y: str, Xs: AbstractSet[str]) -> FrozenSet[str]:
    """ Non-redundant subset of Xs that entail the same E[Y|do(Xs)] """
    return frozenset(Xs & G.do(Xs).An(Y))

