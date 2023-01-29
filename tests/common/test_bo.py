#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: October 8, 2021
#
"""test_bo"""


def test_import_BO():
    from bayeso.bo import BO
    import bayeso.bo

    print(BO)
    print(bayeso.bo.BO)

def test_import_BOwGP():
    from bayeso.bo import BOwGP
    import bayeso.bo

    print(BOwGP)
    print(bayeso.bo.BOwGP)
    print(bayeso.bo.bo_w_gp.BOwGP)

def test_import_BOwTP():
    from bayeso.bo import BOwTP
    import bayeso.bo

    print(BOwTP)
    print(bayeso.bo.BOwTP)
    print(bayeso.bo.bo_w_tp.BOwTP)

def test_import_BOwTrees():
    from bayeso.bo import BOwTrees
    import bayeso.bo

    print(BOwTrees)
    print(bayeso.bo.BOwTrees)
    print(bayeso.bo.bo_w_trees.BOwTrees)
