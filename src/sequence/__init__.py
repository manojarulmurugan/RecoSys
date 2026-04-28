"""Sequence-recommendation models for the 500k REES46 sample.

V7 (GRU4Rec) and V8 (SASRec) live here.  Distinct from
``src.two_tower`` because the data layer (chronological per-user
sequences), training loop (sampled softmax over multiple positions),
and model interface (``encode_sequence`` returning a single user
embedding from the last position) are all different.
"""
