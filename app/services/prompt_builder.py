"""根据白名单选项构建万相编辑指令。

前端不能提交自由 Prompt；所有文本片段均来自 catalog。
"""

from __future__ import annotations

from app.catalogs.appearance_options import get_color, get_style
from app.catalogs.region_definitions import get_region_definition
from app.core.config import Settings
from app.core.exceptions import EditValidationError
from app.schemas.edit import EditOperation, RegionType

_PRESERVE_SUFFIX = (
    "除目标部位外，保持人物面部特征、五官、肤色、表情、身体比例、姿态、"
    "背景和原图摄影风格不变，仅修改遮罩区域，保持写实自然。"
)

_REGION_PROMPT_SUFFIX = {
    RegionType.HAT: "帽子戴在头顶，帽檐位于眉毛上方，不要遮挡眼睛、鼻子和嘴巴。",
    RegionType.GLASSES: "眼镜位于双眼位置，镜片自然透明，不要改变眼睛本身和面部特征。",
    RegionType.EARRING: "耳环自然挂在耳垂位置，不要改变耳朵、脸型和发型。",
    RegionType.HAIR: "发型自然贴合头部轮廓，不要改变五官和脸型。",
    RegionType.TOP: "上衣贴合肩部和躯干，不要改变头部、手部和背景。",
    RegionType.PANTS: "裤子贴合腿部和腰部，不要改变上衣、鞋子和背景。",
    RegionType.SHOES: "鞋子贴合脚部并与地面自然接触，不要改变裤子和腿部。",
}


class PromptBuilder:
    """生成受控的万相局部编辑指令。"""

    def __init__(self, settings: Settings) -> None:
        self._max_length = settings.edit_prompt_max_length

    def build(
        self,
        *,
        person_ordinal: int,
        region_type: RegionType,
        operation: EditOperation,
        style_id: str,
        color_id: str,
    ) -> str:
        definition = get_region_definition(region_type)
        if operation not in definition.supported_operations:
            raise EditValidationError(
                detail=f"{definition.label} 不支持 {operation.value} 操作",
            )
        if operation not in {EditOperation.ADD, EditOperation.REPLACE}:
            raise EditValidationError(detail="当前版本仅支持添加和替换操作")
        try:
            style = get_style(region_type, style_id)
            color = get_color(color_id)
        except KeyError as exc:
            raise EditValidationError(
                detail="选择的款式或颜色不在预设选项中",
            ) from exc

        target = f"{color.prompt_fragment}{style.prompt_fragment}"
        if operation is EditOperation.ADD:
            action = f"为画面中从左到右第{person_ordinal}个人物添加{target}"
        else:
            action = (
                f"将画面中从左到右第{person_ordinal}个人物原有的"
                f"{definition.label}替换为{target}"
            )

        prompt = (
            f"{action}。{_PRESERVE_SUFFIX}"
            f"{_REGION_PROMPT_SUFFIX.get(region_type, '')}"
        )
        if len(prompt) > self._max_length:
            raise EditValidationError(
                detail=f"编辑指令超过 {self._max_length} 字符限制",
            )
        return prompt
