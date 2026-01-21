import pytest

from cmab.scm.causal_diagram import CausalDiagram
from cmab.algorithms.pomis.pomis_sets import POMISs, MUCT, IB

from npsem.model import CausalDiagram as NPSEM_CausalDiagram
from npsem.where_do import MUCT as NPSEM_MUCT
from npsem.where_do import IB as NPSEM_IB
from npsem.where_do import POMISs as NPSEM_POMISs

DIAGRAMS = [
    dict(
        name="Diagram 1",
        nodes = ["S", "T", "W", "Z", "X", "Y"],
        directed_edges = [
        ("S", "W"),
        ("T", "X"),
        ("T", "Y"),
        ("W", "Y"),
        ("Z", "X"),
        ("X", "Y"), ],
        bidirected_edges = [
        ("W", "X", "U_WX"),
        ("Z", "Y", "U_ZY")],
        target="Y",
    ),
    dict(
        name="Diagram 2",
        nodes=["X", "Z", "Y"],
        directed_edges=[("X", "Z"), ("Z", "Y")],
        bidirected_edges=[],
        target="Y",
    ),
]

@pytest.fixture(params=DIAGRAMS, ids=lambda s: s["name"])
def diagrams(request):
    spec = request.param

    npsem_cd = NPSEM_CausalDiagram(
        vs=spec["nodes"],
        directed_edges=spec["directed_edges"],
        bidirected_edges=spec["bidirected_edges"],
    )

    my_cd = CausalDiagram(
        nodes=spec["nodes"],  
        directed_edges=spec["directed_edges"],
        bidirected_edges=spec["bidirected_edges"],
    )

    return npsem_cd, my_cd, spec["target"]


@pytest.mark.parametrize(
    "npsem_fn,my_fn",
    [
        (NPSEM_MUCT, MUCT),
        (NPSEM_IB, IB),
        (NPSEM_POMISs, POMISs),
    ],
)
def test_sets_match_npsem(diagrams, npsem_fn, my_fn):
    npsem_cd, my_cd, target = diagrams
    expected = npsem_fn(npsem_cd, target)
    actual = my_fn(my_cd, target)
    assert expected == actual
