"""可编辑部位配置。

部位名称、依赖顺序、默认 Mask 策略均在配置中维护，业务代码不硬编码部位名称。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.schemas.edit import EditOperation, RegionType


@dataclass(frozen=True, slots=True)
class RegionDefinition:
    region_type: RegionType
    label: str
    supported_operations: tuple[EditOperation, ...]
    generation_order: int
    dependencies: tuple[RegionType, ...] = field(default_factory=tuple)
    object_mask_min_area_ratio: float = 0.0005
    object_mask_max_area_ratio: float = 0.45
    placement_mask_min_area_ratio: float = 0.0003
    placement_mask_max_area_ratio: float = 0.35


REGION_DEFINITIONS: dict[RegionType, RegionDefinition] = {
    RegionType.HAIR: RegionDefinition(
        region_type=RegionType.HAIR,
        label="头发",
        supported_operations=(EditOperation.REPLACE, EditOperation.ADD),
        generation_order=10,
        object_mask_max_area_ratio=0.35,
    ),
    RegionType.HAT: RegionDefinition(
        region_type=RegionType.HAT,
        label="帽子",
        supported_operations=(EditOperation.ADD, EditOperation.REPLACE),
        generation_order=20,
        dependencies=(RegionType.HAIR,),
        object_mask_max_area_ratio=0.25,
    ),
    RegionType.GLASSES: RegionDefinition(
        region_type=RegionType.GLASSES,
        label="眼镜",
        supported_operations=(EditOperation.ADD, EditOperation.REPLACE),
        generation_order=30,
        object_mask_max_area_ratio=0.12,
        placement_mask_max_area_ratio=0.12,
    ),
    RegionType.EARRING: RegionDefinition(
        region_type=RegionType.EARRING,
        label="耳环",
        supported_operations=(EditOperation.ADD, EditOperation.REPLACE),
        generation_order=40,
        object_mask_max_area_ratio=0.06,
        placement_mask_max_area_ratio=0.06,
    ),
    RegionType.TOP: RegionDefinition(
        region_type=RegionType.TOP,
        label="上衣",
        supported_operations=(EditOperation.ADD, EditOperation.REPLACE),
        generation_order=50,
        object_mask_max_area_ratio=0.45,
        placement_mask_max_area_ratio=0.4,
    ),
    RegionType.PANTS: RegionDefinition(
        region_type=RegionType.PANTS,
        label="裤子",
        supported_operations=(EditOperation.ADD, EditOperation.REPLACE),
        generation_order=60,
        object_mask_max_area_ratio=0.45,
        placement_mask_max_area_ratio=0.4,
    ),
    RegionType.SHOES: RegionDefinition(
        region_type=RegionType.SHOES,
        label="鞋子",
        supported_operations=(EditOperation.ADD, EditOperation.REPLACE),
        generation_order=70,
        dependencies=(RegionType.PANTS,),
        object_mask_max_area_ratio=0.25,
    ),
}


def get_region_definition(region_type: RegionType) -> RegionDefinition:
    return REGION_DEFINITIONS[region_type]


def get_generation_order(region_type: RegionType) -> int:
    return REGION_DEFINITIONS[region_type].generation_order
