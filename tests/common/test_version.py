#
# author: Jungtaek Kim (jungtaek.kim.mail@gmail.com)
# last updated: April 30, 2024
#
"""test_import"""


STR_VERSION = '0.6.1'

def test_version_bayeso():
    import bayeso
    assert bayeso.__version__ == STR_VERSION

def test_version_setup():
    import importlib
    assert importlib.metadata.version("bayeso") == STR_VERSION
