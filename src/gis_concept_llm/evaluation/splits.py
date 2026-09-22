"""Leakage-resistant split helpers for GIS concept generalization experiments."""

from __future__ import annotations

from collections.abc import Iterable

from ..schema import ConceptExample


def split_unseen_cities(examples: Iterable[ConceptExample], held_out_cities: set[str]) -> tuple[list[ConceptExample], list[ConceptExample]]:
    """Return train/test partitions with held-out cities exclusively in test.

    Synthetic records remain in train because they have no city identity. Real
    GIS records must carry a city to participate in this split.
    """
    train, test = [], []
    for example in examples:
        if example.source_type == "real_gis" and not example.city:
            raise ValueError(f"real GIS example {example.id} is missing city provenance")
        (test if example.city in held_out_cities else train).append(example)
    train_cities = {example.city for example in train if example.city}
    test_cities = {example.city for example in test if example.city}
    if train_cities & test_cities:
        raise AssertionError("city leakage between train and test")
    return train, test
