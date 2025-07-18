#
# author: Jungtaek Kim (jungtaek.kim.mail@gmail.com)
# last updated: October 13, 2021
#
"""These files are written to implement tree-based regression models.

(i) Breiman, L. (1996). Bagging Predictors. Machine Learning, 24(2),\
pp. 123--140.

(ii) Dietterich, T. G. (2000). An Experimental Comparison of Three Methods\
for Constructing Ensembles of Decision Trees: Bagging, Boosting, and\
Randomization. Machine Learning, 40(2), pp. 139--157.

(iii) Breiman, L. (2001). Random Forests. Machine Learning, 45(1),\
pp. 5--32."""

from bayeso.trees import trees_generic_trees
from bayeso.trees import trees_random_forest


get_generic_trees = trees_generic_trees.get_generic_trees
get_random_forest = trees_random_forest.get_random_forest
