CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

# Model Constants
IGNORE_INDEX = -100

IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
IMAGE_PLACEHOLDER = "<image-placeholder>"

# ======================================================================================================
DEFAULT_VIDEO_TOKEN = "<video>"
DEFAULT_VIDEO_PATCH_TOKEN = "<im_patch>"
DEFAULT_VID_START_TOKEN = "<vid_start>"
DEFAULT_VID_END_TOKEN = "<vid_end>"
VIDEO_PLACEHOLDER = "<video-placeholder>"
# ======================================================================================================

MAX_IMAGE_LENGTH = 16
MAX_VIDEO_LENGTH = 1  # current video datasets only have 1 video?

PAD_LENGTH = 620


# ======================================================================================================
# Qwen2-VL Constants
QWEN2VL_SYSTEM_MESSAGE = "You are a helpful assistant."
QWEN2VL_IM_START_TOKEN = "<|im_start|>"
QWEN2VL_IM_END_TOKEN = "<|im_end|>"
QWEN2VL_IMAGE_TOKEN = "<|image_pad|>"
QWEN2VL_VIDEO_TOKEN = "<|video_pad|>"
QWEN2VL_VISION_START_TOKEN = "<|vision_start|>"
QWEN2VL_VISION_END_TOKEN = "<|vision_end|>"