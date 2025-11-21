from pathlib import Path
from PIL import Image
import numpy as np
import torch

def compute_mean_std(root_dir):
    root = Path(root_dir)

    # .jpg, .jpeg, .JPG, .JPEG 모두 포함
    exts = ["*.jpg", "*.jpeg", "*.JPG", "*.JPEG"]
    image_paths = []
    for ext in exts:
        image_paths.extend(root.rglob(ext))

    if len(image_paths) == 0:
        raise ValueError("No jpg/jpeg images found in the given directory.")

    # 채널별 합과 제곱합을 누적
    sum_c = torch.zeros(3, dtype=torch.float64)
    sum_sq_c = torch.zeros(3, dtype=torch.float64)
    n_pixels_total = 0  # 전체 픽셀 수 (H*W*N)

    for path in image_paths:
        img = Image.open(path).convert("RGB")   # (H, W, 3)
        np_img = np.array(img, dtype=np.float32) / 255.0   # [0,1] 스케일

        # (H*W, 3)로 펼치기
        tensor = torch.from_numpy(np_img).view(-1, 3)  # shape: (P, 3), P = H*W

        # 채널별 합 / 제곱합
        sum_c += tensor.sum(dim=0, dtype=torch.float64)
        sum_sq_c += (tensor ** 2).sum(dim=0, dtype=torch.float64)
        n_pixels_total += tensor.shape[0]

    # 채널별 평균: μ_c = (1 / (N_pixels)) * Σ x
    mean = sum_c / n_pixels_total

    # 채널별 분산: σ^2_c = E[x^2] - (E[x])^2
    var = (sum_sq_c / n_pixels_total) - mean ** 2
    std = torch.sqrt(var)

    return mean, std


if __name__ == "__main__":
    root_dir = "agi/cropped_aligned"  # 여기만 바꿔서 사용
    mean, std = compute_mean_std(root_dir)
    print("Mean (R, G, B):", mean)
    print("Std  (R, G, B):", std)