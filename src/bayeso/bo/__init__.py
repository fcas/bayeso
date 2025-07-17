#
# author: Jungtaek Kim (jungtaek.kim.mail@gmail.com)
# last updated: July 17, 2025
#
"""These files are for implementing Bayesian optimization classes.

(i) Garnett, R. (2023). Bayesian Optimization. Cambridge University Press."""

from bayeso.bo import bo_w_gp
from bayeso.bo import bo_w_tp
from bayeso.bo import bo_w_trees


BO = bo_w_gp.BOwGP
BOwGP = bo_w_gp.BOwGP
BOwTP = bo_w_tp.BOwTP
BOwTrees = bo_w_trees.BOwTrees
