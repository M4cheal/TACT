import numpy as np

def normalize_np(img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_value=255.0):
    """
    对输入图像做 NumPy 版的 ToTensor + Normalize 操作：
      1. 将像素缩放到 [0, 1]
      2. 按通道减去 mean，再除以 std

    支持 img.shape 为 (B, 3, H, W) 或 (3, H, W)。

    参数:
      img       : np.ndarray，dtype 可为 uint8/float32，像素范围 [0, max_value]
      mean      : 3 元组，PyTorch Normalize 的均值
      std       : 3 元组，PyTorch Normalize 的标准差
      max_value : 原始像素的最大值，默认为 255.0

    返回:
      np.ndarray，dtype float32，已做归一化，形状不变
    """
    # 转 float 并缩放到 [0,1]
    img = img.astype(np.float32) / max_value

    # 构造形状为 (3,1,1) 的 mean/std，用于广播
    mean = np.array(mean, dtype=np.float32)[:, None, None]
    std = np.array(std, dtype=np.float32)[:, None, None]

    if img.ndim == 4:
        # (B, 3, H, W) -> 直接在通道维度广播
        return (img - mean[None, ...]) / std[None, ...]
    elif img.ndim == 3:
        # (3, H, W)
        return (img - mean) / std
    else:
        raise ValueError(f"Unsupported img.ndim = {img.ndim}, expected 3 or 4.")



def normalize_image(img_npy):
    """
    :param img_npy: b, c, h, w
    """
    for b in range(img_npy.shape[0]):
        for c in range(img_npy.shape[1]):
            img_npy[b, c] = (img_npy[b, c] - img_npy[b, c].mean()) / img_npy[b, c].std()
    return img_npy


def normalize_image_to_0_1(img):
    if len(img.shape) == 4:
        for b in range(img.shape[0]):
            img[b] = (img[b]-img[b].min())/(img[b].max()-img[b].min())
    else:
        img = (img-img.min())/(img.max()-img.min())
    return img


def normalize_image_to_m1_1(img):
    return -1 + 2 * (img-img.min())/(img.max()-img.min())


# def normalize_image_to_0_1(img):
#     for b in range(img.shape[0]):
#         img[b] = (img[b]-img[b].min())/(img[b].max()-img[b].min())
#     return img
#
#
# def normalize_image_to_m1_1(img):
#     for b in range(img.shape[0]):
#         img[b] = -1 + 2 * (img[b]-img[b].min())/(img[b].max()-img[b].min())
#     return img
