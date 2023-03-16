import numpy as np
import sys

import gradgen

sys.path.insert(1, '../../../clone_raocp-toolbox/raocp-toolbox')
import raocp.core as core

p = np.array([[0.5, 0.5],
              [0.5, 0.5]])
v = np.array([0.5, 0.5])
(N, tau) = (2, 2)
markov_tree = core.MarkovChainScenarioTreeFactory(p, v, N, tau).create()

markov_tree.bulls_eye_plot(dot_size=6, radius=300, filename='scenario-tree.eps')
print(sum(markov_tree.probability_of_node(markov_tree.nodes_at_stage(2))))
print(markov_tree.probability_of_node(markov_tree.nodes_at_stage(1)))
print(markov_tree)

gradgen.CostGradientStochastic()
