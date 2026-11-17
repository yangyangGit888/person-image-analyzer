"""外观编辑白名单配置。

前端只能提交这里的 option_id，不能提交自由文本或任意 Prompt。
"""

from __future__ import annotations

from dataclasses import dataclass

from app.catalogs.region_definitions import REGION_DEFINITIONS
from app.schemas.catalog import (
    AppearanceOptionsData,
    ColorOption,
    OptionItem,
    RegionOptionGroup,
)
from app.schemas.edit import RegionType


@dataclass(frozen=True, slots=True)
class OptionDefinition:
    option_id: str
    label: str
    prompt_fragment: str


COLOR_DEFINITIONS: tuple[OptionDefinition, ...] = (
    OptionDefinition("black", "黑色", "黑色"),
    OptionDefinition("white", "白色", "白色"),
    OptionDefinition("red", "红色", "红色"),
    OptionDefinition("blue", "蓝色", "蓝色"),
    OptionDefinition("navy", "深蓝色", "深蓝色"),
    OptionDefinition("gray", "灰色", "灰色"),
    OptionDefinition("beige", "米色", "米色"),
    OptionDefinition("brown", "棕色", "棕色"),
    OptionDefinition("pink", "粉色", "粉色"),
    OptionDefinition("green", "绿色", "绿色"),
    OptionDefinition("yellow", "黄色", "黄色"),
    OptionDefinition("purple", "紫色", "紫色"),
    OptionDefinition("silver", "银色", "银色"),
    OptionDefinition("gold", "金色", "金色"),
    OptionDefinition("transparent", "透明", "透明"),
)


STYLE_DEFINITIONS: dict[RegionType, tuple[OptionDefinition, ...]] = {
    RegionType.HAT: (
        OptionDefinition("baseball_cap", "棒球帽", "棒球帽"),
        OptionDefinition("bucket_hat", "渔夫帽", "渔夫帽"),
        OptionDefinition("beanie", "针织帽", "针织帽"),
        OptionDefinition("beret", "贝雷帽", "贝雷帽"),
        OptionDefinition("flat_cap", "鸭舌帽", "鸭舌帽"),
    ),
    RegionType.GLASSES: (
        OptionDefinition("metal_round", "金属圆框眼镜", "金属圆框眼镜"),
        OptionDefinition("square_frame", "方框眼镜", "方框眼镜"),
        OptionDefinition("rimless", "无框眼镜", "无框眼镜"),
        OptionDefinition("transparent_frame", "透明框眼镜", "透明框眼镜"),
        OptionDefinition("sunglasses", "墨镜", "墨镜"),
    ),
    RegionType.HAIR: (
        OptionDefinition("short", "短发", "短发"),
        OptionDefinition("long_straight", "长直发", "长直发"),
        OptionDefinition("curly", "卷发", "卷发"),
        OptionDefinition("wavy", "波浪发", "波浪发"),
        OptionDefinition("bob", "波波头", "波波头"),
        OptionDefinition("ponytail", "马尾", "马尾"),
    ),
    RegionType.EARRING: (
        OptionDefinition("stud", "耳钉", "耳钉"),
        OptionDefinition("hoop", "圆环耳环", "圆环耳环"),
        OptionDefinition("drop", "水滴耳环", "水滴耳环"),
        OptionDefinition("pearl", "珍珠耳环", "珍珠耳环"),
        OptionDefinition("tassel", "流苏耳环", "流苏耳环"),
    ),
    RegionType.TOP: (
        OptionDefinition("tshirt", "T恤", "T恤"),
        OptionDefinition("shirt", "衬衫", "衬衫"),
        OptionDefinition("hoodie", "卫衣", "卫衣"),
        OptionDefinition("sweater", "毛衣", "毛衣"),
        OptionDefinition("blazer", "西装", "西装"),
        OptionDefinition("jacket", "夹克", "夹克"),
    ),
    RegionType.PANTS: (
        OptionDefinition("jeans", "牛仔裤", "牛仔裤"),
        OptionDefinition("slacks", "西裤", "西裤"),
        OptionDefinition("joggers", "运动裤", "运动裤"),
        OptionDefinition("cargo", "工装裤", "工装裤"),
        OptionDefinition("leggings", "紧身裤", "紧身裤"),
        OptionDefinition("shorts", "短裤", "短裤"),
    ),
    RegionType.SHOES: (
        OptionDefinition("sneakers", "运动鞋", "运动鞋"),
        OptionDefinition("leather", "皮鞋", "皮鞋"),
        OptionDefinition("boots", "靴子", "靴子"),
        OptionDefinition("sandals", "凉鞋", "凉鞋"),
        OptionDefinition("loafers", "乐福鞋", "乐福鞋"),
        OptionDefinition("heels", "高跟鞋", "高跟鞋"),
    ),
}


STYLE_LOOKUP: dict[RegionType, dict[str, OptionDefinition]] = {
    region: {item.option_id: item for item in items}
    for region, items in STYLE_DEFINITIONS.items()
}
COLOR_LOOKUP: dict[str, OptionDefinition] = {
    item.option_id: item for item in COLOR_DEFINITIONS
}


def get_style(region_type: RegionType, style_id: str) -> OptionDefinition:
    style = STYLE_LOOKUP.get(region_type, {}).get(style_id)
    if style is None:
        raise KeyError(style_id)
    return style


def get_color(color_id: str) -> OptionDefinition:
    color = COLOR_LOOKUP.get(color_id)
    if color is None:
        raise KeyError(color_id)
    return color


def build_options_data() -> AppearanceOptionsData:
    regions: list[RegionOptionGroup] = []
    for region_type in (
        RegionType.HAT,
        RegionType.GLASSES,
        RegionType.HAIR,
        RegionType.EARRING,
        RegionType.TOP,
        RegionType.PANTS,
        RegionType.SHOES,
    ):
        definition = REGION_DEFINITIONS[region_type]
        regions.append(
            RegionOptionGroup(
                region_type=region_type,
                label=definition.label,
                supported_operations=list(definition.supported_operations),
                styles=[
                    OptionItem(
                        option_id=item.option_id,
                        label=item.label,
                        prompt_fragment=item.prompt_fragment,
                    )
                    for item in STYLE_DEFINITIONS[region_type]
                ],
                colors=[
                    ColorOption(
                        option_id=item.option_id,
                        label=item.label,
                        prompt_fragment=item.prompt_fragment,
                    )
                    for item in COLOR_DEFINITIONS
                ],
            )
        )
    return AppearanceOptionsData(
        color_options=[
            ColorOption(
                option_id=item.option_id,
                label=item.label,
                prompt_fragment=item.prompt_fragment,
            )
            for item in COLOR_DEFINITIONS
        ],
        regions=regions,
    )
