#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: October 13, 2021
#
"""These files are written to implement tree-based regression models."""

from bayeso.trees import trees_generic_trees
from bayeso.trees import trees_random_forest


get_generic_trees = trees_generic_trees.get_generic_trees
get_random_forest = trees_random_forest.get_random_forest
