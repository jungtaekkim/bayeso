#
# author: Jungtaek Kim (jungtaek.kim.mail@gmail.com)
# last updated: January 4, 2024
#
"""test_import"""


STR_VERSION = '0.6.0'

def test_version_bayeso():
    import bayeso
    assert bayeso.__version__ == STR_VERSION

def test_version_setup():
    import pkg_resources
    assert pkg_resources.require("bayeso")[0].version == STR_VERSION
