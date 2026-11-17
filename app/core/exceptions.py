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
    default_detail = "仅支持 JPG、JPEG、PNG、WEBP、HEIC/HEIF 格式的图片"


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


class ResourceNotFoundError(AppError):
    code = "RESOURCE_NOT_FOUND"
    status_code = 404
    default_message = "资源不存在"
    default_detail = "请求的图片、人物、区域或任务不存在"


class EditConflictError(AppError):
    code = "EDIT_CONFLICT"
    status_code = 409
    default_message = "编辑方案冲突"
    default_detail = "当前编辑操作与已有状态冲突"


class EditValidationError(AppError):
    code = "EDIT_VALIDATION_ERROR"
    status_code = 422
    default_message = "编辑参数不合法"
    default_detail = "编辑部位、操作或选项不合法"


class StorageError(AppError):
    code = "STORAGE_ERROR"
    status_code = 500
    default_message = "存储服务异常"
    default_detail = "无法保存或读取图片与 Mask"


class SegmentationUnavailableError(AppError):
    code = "SEGMENTATION_UNAVAILABLE"
    status_code = 503
    default_message = "分割服务不可用"
    default_detail = "本地分割模型未安装或未正确配置"


class SegmentationError(AppError):
    code = "SEGMENTATION_ERROR"
    status_code = 502
    default_message = "图像分割失败"
    default_detail = "无法从图片中分割出有效区域"


class MaskGenerationError(AppError):
    code = "MASK_GENERATION_ERROR"
    status_code = 422
    default_message = "Mask 生成失败"
    default_detail = "无法为该部位生成有效的编辑 Mask"


class RegionDetectionError(AppError):
    code = "REGION_DETECTION_ERROR"
    status_code = 502
    default_message = "部位识别失败"
    default_detail = "模型未能返回可用的部位定位结果"


class ImageEditConfigError(AppError):
    code = "IMAGE_EDIT_CONFIG_ERROR"
    status_code = 503
    default_message = "图像编辑配置错误"
    default_detail = "未配置 DASHSCOPE_API_KEY 或万相服务参数"


class ImageEditAuthError(AppError):
    code = "IMAGE_EDIT_AUTH_ERROR"
    status_code = 502
    default_message = "图像编辑失败"
    default_detail = "万相服务鉴权失败，请检查 DASHSCOPE_API_KEY 与地域"


class ImageEditTimeoutError(AppError):
    code = "IMAGE_EDIT_TIMEOUT"
    status_code = 504
    default_message = "图像编辑超时"
    default_detail = "万相编辑任务在规定时间内未完成"


class ImageEditFailedError(AppError):
    code = "IMAGE_EDIT_FAILED"
    status_code = 502
    default_message = "图像编辑失败"
    default_detail = "万相图像编辑任务执行失败"


class ImageEditContentRejectedError(AppError):
    code = "IMAGE_EDIT_CONTENT_REJECTED"
    status_code = 422
    default_message = "图像编辑被拒绝"
    default_detail = "图片或编辑指令未通过内容安全审核"


class ImageEditInvalidOutputError(AppError):
    code = "IMAGE_EDIT_INVALID_OUTPUT"
    status_code = 502
    default_message = "图像编辑结果无效"
    default_detail = "万相返回的图片无法解析或尺寸不合法"


class ImageEditRateLimitError(AppError):
    code = "IMAGE_EDIT_RATE_LIMITED"
    status_code = 503
    default_message = "图像编辑请求过于频繁"
    default_detail = "万相服务限流，请稍后重试"
