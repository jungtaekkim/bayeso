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
