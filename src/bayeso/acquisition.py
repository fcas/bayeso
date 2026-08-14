#
# author: Jungtaek Kim (jungtaek.kim.mail@gmail.com)
# last updated: July 17, 2025
#
"""This file is not only to make these acquisitions
functions compatible with earlier versions but also
to shorten their names for simplicity."""
# TODO: Keep this file for a while.

from bayeso.acquisitions import probability_of_improvement
from bayeso.acquisitions import expected_improvement
from bayeso.acquisitions import log_expected_improvement
from bayeso.acquisitions import gaussian_process_upper_confidence_bound
from bayeso.acquisitions import augmented_expected_improvement
from bayeso.acquisitions import pure_exploitation
from bayeso.acquisitions import pure_exploration


pi = probability_of_improvement.acq_fun
ei = expected_improvement.acq_fun
log_ei = log_expected_improvement.acq_fun
ucb = gaussian_process_upper_confidence_bound.acq_fun
aei = augmented_expected_improvement.acq_fun
pure_exploit = pure_exploitation.acq_fun
pure_explore = pure_exploration.acq_fun
