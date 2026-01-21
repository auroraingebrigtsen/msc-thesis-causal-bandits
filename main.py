from cmab.scm.intervention_domain.interval import IntervalInterventionDomain
from cmab.scm.pmf.bernoulli import BernoulliPmf
from cmab.scm.mechanism.linear import LinearMechanism
from cmab.scm.scm import SCM
from cmab.environments import CausalBanditEnv, NSCausalBanditEnv
from cmab.algorithms.ucb import UCBAgent, SlidingWindowUCBAgent
from cmab.utils.plotting import  plot_regrets
from cmab.metrics.cumulative_regret import CumulativeRegret
import numpy as np
from cmab.algorithms.ucb.pomis_kl_ucb import PomisKLUCBAgent
from cmab.scm.causal_diagram import CausalDiagram
from cmab.algorithms.pomis.pomis_sets import POMISs, MUCT, IB

from npsem.model import CausalDiagram as NPSEM_CausalDiagram
from npsem.where_do import MUCT as NPSEM_MUCT
from npsem.where_do import IB as NPSEM_IB
from npsem.where_do import POMISs as NPSEM_POMISs

def main():

    ### FROM GITHUB NPSEM PACKAGE
    npsem_cd = NPSEM_CausalDiagram(
        vs=['S', 'T', 'W', 'Z', 'X', 'Y'],
        directed_edges=[('S', 'W'),
               ('T', 'X'),
               ('T', 'Y'),
               ('W', 'Y'),
               ('Z', 'X'),
               ('X', 'Y',)],    
        bidirected_edges=[('W', 'X', 'U_WX'), ('Z', 'Y', 'U_ZY')]
    )

    npsem_muct = NPSEM_MUCT(npsem_cd, 'Y')
    print(f"NPSEM MUCT: {npsem_muct}")

    npsem_ib = IB(npsem_cd, 'Y')
    print(f"NPSEM IB: {npsem_ib}")

    npsem_pomis = NPSEM_POMISs(npsem_cd, 'Y')
    print(f"NPSEM POMISs: {npsem_pomis}")


    ### MY IMPLEMENTATION
    causal_diagram = CausalDiagram(
        nodes=frozenset({'S', 'T', 'W', 'Z', 'X', 'Y'}),
        directed_edges=[('S', 'W'),
               ('T', 'X'),
               ('T', 'Y'),
               ('W', 'Y'),
               ('Z', 'X'),
               ('X', 'Y',)],
        bidirected_edges=[('W', 'X', 'U_WX'), ('Z', 'Y', 'U_ZY')]
    )

    muct = MUCT(causal_diagram, 'Y')
    print(f"MUCT: {muct}")

    ib = IB(causal_diagram, 'Y')
    print(f"IB: {ib}")

    pomis = POMISs(causal_diagram, 'Y')
    print(f"POMISs: {pomis}")

if __name__ == "__main__":
    main()