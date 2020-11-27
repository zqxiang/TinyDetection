from tinydet.utils.registry import Registry
from .base import Backbone

__all__ = ["BACKBONE_REGISTRY", "build_backbone"]

BACKBONE_REGISTRY = Registry("BACKBONE")


def build_backbone(cfg) -> Backbone:
    return BACKBONE_REGISTRY.get(cfg.MODEL.BACKBONE.NAME)
