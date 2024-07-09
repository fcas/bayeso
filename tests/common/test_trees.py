#
# author: Jungtaek Kim (jungtaek.kim.mail@gmail.com)
# last updated: July 9, 2024
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
