#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: June 24, 2021
#
"""test_import"""


STR_VERSION = '0.5.1'

def test_version_bayeso():
    import bayeso
    assert bayeso.__version__ == STR_VERSION

def test_version_setup():
    import pkg_resources
    assert pkg_resources.require("bayeso")[0].version == STR_VERSION
