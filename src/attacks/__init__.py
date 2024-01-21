"""Implementations of the two attacks as aggregators with their corresponding client datasets."""

from .backdoor_attack import (get_backdoor_agg, BackdoorDataset,
                              BACKDOOR_TRIGGERS, BACKDOOR_TARGETS)
from .fairness_attack import (get_unfair_fedavg_agg, UnfairDataset,
                              UNFAIR_ATTRIBUTE, UNFAIR_MODIFICATION)

ATTACKS = {
    "backdoor_attack": get_backdoor_agg,
    "fairness_attack_fedavg": get_unfair_fedavg_agg
}
