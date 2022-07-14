# -*- coding: utf-8 -*-

from .aggregator import SumAggregator, ConcatAggregator, NeighAggregator, SingleAggregator, newAggregator

Aggregator = {
    'sum': SumAggregator,
    'concat': ConcatAggregator,
    'neigh': NeighAggregator,
    'single': SingleAggregator,
    'min_ent' : newAggregator
}
