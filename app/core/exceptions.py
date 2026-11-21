"""统一业务异常定义。

所有异常都携带稳定的错误码和对外安全的错误描述。异常信息中禁止放入 API
Key、内部环境变量或完整请求头。
"""

from __future__ import annotations


class AppError(Exception):
    """业务异常基类。"""

    code: str = "APP_ERROR"
    status_code: int = 500
    default_message: str = "服务处理失败"
    default_detail: str = "发生未知错误"

    def __init__(
        self,
        detail: str | None = None,
        *,
        message: str | None = None,
        status_code: int | None = None,
        code: str | None = None,
    ) -> None:
        self.detail = detail or self.default_detail
        self.message = message or self.default_message
        if status_code is not None:
            self.status_code = status_code
        if code is not None:
            self.code = code
        super().__init__(self.detail)


class ConfigurationError(AppError):
    code = "APP_CONFIG_ERROR"
    status_code = 500
    default_message = "服务配置错误"
    default_detail = "服务器配置不完整"


class EmptyFileError(AppError):
    code = "EMPTY_FILE"
    status_code = 400
    default_message = "上传文件为空"
    default_detail = "上传的图片文件为空"


class ImageTooLargeError(AppError):
    code = "IMAGE_TOO_LARGE"
    status_code = 413
    default_message = "图片过大"
    default_detail = "图片大小超过服务限制"


class UnsupportedImageFormatError(AppError):
    code = "UNSUPPORTED_IMAGE_FORMAT"
    status_code = 415
    default_message = "图片格式不支持"
    default_detail = "仅支持 JPG、JPEG、PNG、GIF、WEBP、HEIC/HEIF 格式的图片"


class InvalidImageError(AppError):
    code = "INVALID_IMAGE"
    status_code = 400
    default_message = "图片文件无效"
    default_detail = "图片文件已损坏或不是合法的图片"


class ImageProcessingError(AppError):
    code = "IMAGE_PROCESSING_ERROR"
    status_code = 400
    default_message = "图片处理失败"
    default_detail = "无法处理上传的图片"


class NoPersonDetectedError(AppError):
    code = "NO_PERSON_DETECTED"
    status_code = 422
    default_message = "图片分析失败"
    default_detail = "图片中未检测到人物"


class AIConfigError(AppError):
    code = "AI_CONFIG_ERROR"
    status_code = 500
    default_message = "AI 模型配置错误"
    default_detail = "AI 模型配置不完整或不合法"


class AIAuthenticationError(AppError):
    code = "AI_AUTH_ERROR"
    status_code = 502
    default_message = "图片分析失败"
    default_detail = "模型服务鉴权失败，请检查 api key 配置"


class AITimeoutError(AppError):
    code = "AI_TIMEOUT"
    status_code = 504
    default_message = "图片分析超时"
    default_detail = "模型服务请求超时，请稍后重试"


class AIRateLimitError(AppError):
    code = "AI_RATE_LIMITED"
    status_code = 503
    default_message = "图片分析失败"
    default_detail = "模型服务请求过于频繁，请稍后重试"


class AIServiceUnavailableError(AppError):
    code = "AI_SERVICE_UNAVAILABLE"
    status_code = 503
    default_message = "图片分析失败"
    default_detail = "模型服务暂时不可用，请稍后重试"


class AIAPIError(AppError):
    code = "AI_API_ERROR"
    status_code = 502
    default_message = "图片分析失败"
    default_detail = "模型调用失败"


class AIModelNotSupportVisionError(AppError):
    code = "AI_MODEL_NOT_SUPPORT_VISION"
    status_code = 502
    default_message = "图片分析失败"
    default_detail = "当前模型不支持图片输入，请更换支持视觉的模型"


class AIEmptyResponseError(AppError):
    code = "AI_EMPTY_RESPONSE"
    status_code = 502
    default_message = "图片分析失败"
    default_detail = "模型返回内容为空，可能是输出被截断，请重试"


class AIInvalidJSONError(AppError):
    code = "AI_INVALID_JSON"
    status_code = 502
    default_message = "图片分析失败"
    default_detail = "模型返回内容不是合法 json"


class AIInvalidResponseError(AppError):
    code = "AI_INVALID_RESPONSE"
    status_code = 502
    default_message = "图片分析失败"
    default_detail = "模型返回结果缺少必要字段或格式不正确"


class InternalError(AppError):
    code = "INTERNAL_ERROR"
    status_code = 500
    default_message = "服务器内部错误"
    default_detail = "服务器处理请求时发生异常"


# ---------- Prompt / 配置 ----------
class PromptConfigError(AppError):
    code = "PROMPT_CONFIG_ERROR"
    status_code = 500
    default_message = "Prompt 配置错误"
    default_detail = "Prompt 文件缺失或格式不正确"


# ---------- 存储 / 资源 ----------
class StorageError(AppError):
    code = "STORAGE_ERROR"
    status_code = 500
    default_message = "本地存储失败"
    default_detail = "无法读写本地存储目录"


class ResourceNotFoundError(AppError):
    code = "RESOURCE_NOT_FOUND"
    status_code = 404
    default_message = "资源不存在"
    default_detail = "请求的资源不存在或已被清理"


class AnalysisNotFoundError(ResourceNotFoundError):
    default_detail = "分析记录不存在或已过期"


class ItemNotFoundError(ResourceNotFoundError):
    default_detail = "物品不存在或不属于当前分析结果"


class TaskNotFoundError(ResourceNotFoundError):
    default_detail = "视频任务不存在"


# ---------- 图片 / 坐标 ----------
class ImageNotAvailableError(AppError):
    code = "IMAGE_NOT_AVAILABLE"
    status_code = 410
    default_message = "原图不可用"
    default_detail = "原图未保存或已超过保留期限"


class InvalidBBoxError(AppError):
    code = "INVALID_BBOX"
    status_code = 422
    default_message = "坐标无效"
    default_detail = "Bounding Box 缺失或超出图片边界"


# ---------- 视频 ----------
class InvalidVideoError(AppError):
    code = "INVALID_VIDEO"
    status_code = 400
    default_message = "视频文件无效"
    default_detail = "视频文件损坏或无法解析"


class UnsupportedVideoFormatError(AppError):
    code = "UNSUPPORTED_VIDEO_FORMAT"
    status_code = 415
    default_message = "视频格式不支持"
    default_detail = "仅支持常见容器格式的 H.264/H.265/VP9/AV1 视频"


class VideoTooLargeError(AppError):
    code = "VIDEO_TOO_LARGE"
    status_code = 413
    default_message = "视频过大"
    default_detail = "视频文件超过大小限制"


class VideoTooLongError(AppError):
    code = "VIDEO_TOO_LONG"
    status_code = 422
    default_message = "视频过长"
    default_detail = "视频时长超过限制"


class VideoResolutionError(AppError):
    code = "VIDEO_RESOLUTION_INVALID"
    status_code = 422
    default_message = "视频分辨率不支持"
    default_detail = "视频分辨率超出允许范围"


class VideoDecodeError(AppError):
    code = "VIDEO_DECODE_FAILED"
    status_code = 422
    default_message = "视频解码失败"
    default_detail = "无法从视频中解码出有效画面"


class FrameExtractionError(AppError):
    code = "FRAME_EXTRACTION_FAILED"
    status_code = 500
    default_message = "抽帧失败"
    default_detail = "无法从视频中提取候选帧"


class NoValidFramesError(AppError):
    code = "NO_VALID_FRAMES"
    status_code = 422
    default_message = "没有有效画面"
    default_detail = "视频中没有清晰可用的画面"


class UnsupportedMultiPersonError(AppError):
    code = "UNSUPPORTED_MULTI_PERSON"
    status_code = 422
    default_message = "不支持多人视频"
    default_detail = "当前版本仅支持单人短视频"


# ---------- 任务 ----------
class TaskNotCompletedError(AppError):
    code = "TASK_NOT_COMPLETED"
    status_code = 409
    default_message = "任务尚未完成"
    default_detail = "视频任务仍在处理中，请稍后查询"


class TaskCancelledError(AppError):
    code = "TASK_CANCELLED"
    status_code = 409
    default_message = "任务已取消"
    default_detail = "视频任务已被取消"


class TaskTimeoutError(AppError):
    code = "TASK_TIMEOUT"
    status_code = 504
    default_message = "任务超时"
    default_detail = "视频任务处理超时"


class TaskFailedError(AppError):
    code = "TASK_FAILED"
    status_code = 409
    default_message = "任务失败"
    default_detail = "视频任务处理失败，请查看任务错误信息"


# ---------- Mask ----------
class ItemHasNoGeometryError(AppError):
    code = "ITEM_HAS_NO_GEOMETRY"
    status_code = 422
    default_message = "物品缺少定位信息"
    default_detail = "该物品没有可靠的 Bounding Box，无法生成 Mask"


class MaskProviderUnavailableError(AppError):
    code = "MASK_PROVIDER_UNAVAILABLE"
    status_code = 503
    default_message = "Mask 能力不可用"
    default_detail = "当前未配置可用的 Mask 生成能力"


class MaskGenerationError(AppError):
    code = "MASK_GENERATION_FAILED"
    status_code = 500
    default_message = "Mask 生成失败"
    default_detail = "生成物品 Mask 时发生异常"


# ---------- 内容安全 ----------
class AIContentRefusedError(AppError):
    code = "AI_CONTENT_REFUSED"
    status_code = 422
    default_message = "模型拒绝处理"
    default_detail = "模型因内容安全策略拒绝处理该图片"
