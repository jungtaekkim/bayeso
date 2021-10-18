#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: October 13, 2021
#
"""test_trees"""


def test_import_get_generic_trees():
    from bayeso.trees import get_generic_trees
    import bayeso.trees

    print(get_generic_trees)
    print(bayeso.trees.get_generic_trees)

def test_import_get_random_forest():
    from bayeso.trees import get_random_forest
    import bayeso.trees

    print(get_random_forest)
    print(bayeso.trees.get_random_forest)
