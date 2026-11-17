"""预设编辑选项的 API 模型。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.edit import EditOperation, RegionType


class CatalogSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")


class OptionItem(CatalogSchema):
    option_id: str
    label: str
    prompt_fragment: str


class ColorOption(CatalogSchema):
    option_id: str
    label: str
    prompt_fragment: str


class RegionOptionGroup(CatalogSchema):
    region_type: RegionType
    label: str
    supported_operations: list[EditOperation]
    styles: list[OptionItem]
    colors: list[ColorOption]


class AppearanceOptionsData(CatalogSchema):
    color_options: list[ColorOption] = Field(
        default_factory=list,
        description="所有部位通用的颜色白名单",
    )
    regions: list[RegionOptionGroup]


class AppearanceOptionsResponse(CatalogSchema):
    success: Literal[True] = True
    message: str = "ok"
    data: AppearanceOptionsData
