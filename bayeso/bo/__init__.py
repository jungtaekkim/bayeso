#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: October 10, 2021
#
"""These files are for implementing Bayesian optimization classes.
"""

from bayeso.bo import bo_w_gp
from bayeso.bo import bo_w_tp

BO = bo_w_gp.BOwGP
BOwGP = bo_w_gp.BOwGP
BOwTP = bo_w_tp.BOwTP
