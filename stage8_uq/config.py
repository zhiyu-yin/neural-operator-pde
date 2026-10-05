"""Predeclared Stage 8A constants; no seed or protocol selection on evaluation data."""
from dataclasses import replace
from stage6_control.config import Config

BASELINE = 'f095871d842e469f51d49751a7459684a507fd77'
TAG = 'v1.0.0'
D_HASH = 'cf8899453cadf2443961fd65daa736269ed671dd26a375fd3edc1159ae56ca33'
MEMBER_SEEDS = ((2028, 5029), (8101, 8201), (8102, 8202), (8103, 8203), (8104, 8204))
ENSEMBLE_SIZE = 5
ARCHITECTURE = dict(modes=12, width=32, blocks=4, projection_width=64)
OPTIMIZER = dict(lr=5e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
EPSILON_FACTOR = 1e-12


def member_config(member_id: int) -> Config:
    """Only initialization and shuffle seeds differ from Model D's Config."""
    if isinstance(member_id, bool) or member_id not in range(ENSEMBLE_SIZE):
        raise ValueError('Member ID must be 0..4')
    seed, shuffle = MEMBER_SEEDS[member_id]
    return replace(Config(), seed=seed, loader_seed=shuffle)
