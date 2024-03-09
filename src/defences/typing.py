"""Typing definitions for defences."""

from collections.abc import Callable
from flwr.server.strategy import Strategy

from util import Cfg

AggregationWrapper = Callable[[Strategy, int, Cfg, object], Strategy]
