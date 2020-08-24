# test_import
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: August 24, 2020


STR_VERSION = '0.4.3'

def test_version_bayeso():
    import bayeso
    assert bayeso.__version__ == STR_VERSION

def test_version_setup():
    import pkg_resources
    assert pkg_resources.require("bayeso")[0].version == STR_VERSION
