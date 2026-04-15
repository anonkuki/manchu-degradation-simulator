# advanced_degradations_v5.py

import cv2
import numpy as np
import random
from perlin_noise import PerlinNoise # 你可能需要安装这个库：pip install perlin-noise

# 导入新的文字破坏函数
from advanced_text_damage import apply_advanced_text_damage 


def generate_irregular_mask_perlin(shape, scale=100.0, octaves=4, threshold=0.5):
    """(辅助函数) 使用Perlin噪声生成自然、不规则的斑块蒙版。"""
    noise = PerlinNoise(octaves=octaves, seed=random.randint(1, 10000))
    h, w = shape[:2]
    pic = np.array([[noise([i/scale, j/scale]) for j in range(w)] for i in range(h)])
    pic_min, pic_max = np.min(pic), np.max(pic)
    if pic_max > pic_min: pic = (pic - pic_min) / (pic_max - pic_min)
    mask = (pic > threshold).astype(np.uint8) * 255
    return mask

def apply_character_erasure_v5(image, clean_patch, ocr_boxes, background_patch):
    """
    更稳健的 OCR-based 字符擦除实现（v5）。
    - image: 要处理的 BGR numpy 数组（H,W,3）
    - clean_patch: 原始干净 patch（用于参考，H,W,3）
    - ocr_boxes: 列表，每项为 [x1,y1,x2,y2]（相对于 patch 的坐标）
    - background_patch: 用于填充擦除区域的背景 patch（H,W,3）
    返回：处理后的图像（numpy）
    """
    # 拷贝输入，避免原地修改
    out = image.copy()
    h, w = out.shape[:2]

    # 创建擦除 mask（单通道）
    eraser_mask = np.zeros((h, w), dtype=np.uint8)

    # 最大允许的 thickness（保守值）
    MAX_THICKNESS = max(1, min(h, w) // 2)

    for box in ocr_boxes:
        try:
            x1, y1, x2, y2 = map(int, box)
        except Exception:
            continue

        # 安全裁剪边界
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h, y2))
        if x2 <= x1 or y2 <= y1:
            continue

        box_w = x2 - x1
        box_h = y2 - y1

        # 根据 box 大小决定擦除策略
        # 线条数量与厚度随 box 大小变化，但最低 thickness = 1
        num_lines = max(1, int(round(box_w / 20)))  # 经验值：每 ~20px 画一条
        thickness = max(1, int(round(min(box_w, box_h) * 0.12)))  # 经验值：约为 min(box_w,box_h) 的 12%
        thickness = max(1, min(thickness, MAX_THICKNESS))

        for _ in range(num_lines):
            # 线段起点和终点在 box 内随机选择，使擦除效果看起来更真实
            x_start = random.randint(x1, max(x1, x2 - 1))
            y_start = random.randint(y1, max(y1, y2 - 1))
            x_end = random.randint(x1, max(x1, x2 - 1))
            y_end = random.randint(y1, max(y1, y2 - 1))

            # 有一定概率画更长的跨行擦除线（增加多样性）
            if random.random() < 0.2:
                # 横向或纵向延伸一点
                extend_x = int(box_w * random.uniform(0.0, 0.4))
                extend_y = int(box_h * random.uniform(0.0, 0.4))
                x_end = max(0, min(w - 1, x_end + (random.choice([-1, 1]) * extend_x)))
                y_end = max(0, min(h - 1, y_end + (random.choice([-1, 1]) * extend_y)))

            # 确保 thickness 为正整数且不超过 MAX_THICKNESS
            th = max(1, min(thickness, MAX_THICKNESS))

            # 画到擦除 mask 上（白=擦除）
            cv2.line(eraser_mask, (x_start, y_start), (x_end, y_end), 255, th)

    # 用背景图替换擦除区域（白=255）
    mask_inds = eraser_mask == 255
    if np.any(mask_inds):
        # 对三通道替换
        out[mask_inds] = background_patch[mask_inds]

    return out

def apply_paper_damage_v5(image):
    """【纸张损伤 V5】: 实现尺寸/数量负相关，且不规则形状为单一平滑整体。"""
    output_image = image.copy()
    h, w = image.shape[:2]
    total_image_area = h * w
    fill_color = random.choice([[0, 0, 0], [255, 255, 255]])
    shape_type = random.choice(['rectangles', 'blob'])
    if shape_type == 'rectangles':
        target_damage_area = total_image_area * random.uniform(0.05, 0.25)
        current_damage_area = 0
        for _ in range(15):
            if current_damage_area >= target_damage_area: break
            max_rect_w, max_rect_h = int(w * 0.4), int(h * 0.4)
            rect_w = random.randint(20, max_rect_w)
            rect_h = random.randint(20, max_rect_h)
            x1, y1 = random.randint(0, w - rect_w), random.randint(0, h - rect_h)
            cv2.rectangle(output_image, (x1, y1), (x1 + rect_w, y1 + rect_h), fill_color, -1)
            current_damage_area += rect_w * rect_h
    elif shape_type == 'blob':
        sharp_mask = generate_irregular_mask_perlin(image.shape, scale=random.uniform(150, 300), octaves=6, threshold=0.5)
        kernel_size = random.randint(15, 25) * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        closed_mask = cv2.morphologyEx(sharp_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        blur_kernel_size = random.randint(40, 60) * 2 + 1
        smooth_mask = cv2.GaussianBlur(closed_mask, (blur_kernel_size, blur_kernel_size), 0)
        _, final_blob_mask = cv2.threshold(smooth_mask, 127, 255, cv2.THRESH_BINARY)
        output_image[final_blob_mask > 0] = fill_color
    return output_image

# 移除 apply_ink_bleed_v5
# def apply_ink_bleed_v5(image):
#     """【墨水侵蚀 V5】: 效果显著增强，使用中值滤波+强力膨胀，彻底破坏文字结构。"""
#     stain_mask_sharp = generate_irregular_mask_perlin(image.shape, scale=random.uniform(200, 350), octaves=6, threshold=0.55)
#     smooth_mask = cv2.GaussianBlur(stain_mask_sharp.astype(float), (151, 151), 0)
#     max_val = smooth_mask.max()
#     if max_val > 0: smooth_mask = smooth_mask / max_val
#     smooth_mask_3d = np.expand_dims(smooth_mask, axis=2)
#     median_kernel_size = random.randint(12, 18) * 2 + 1
#     bled_image_base = cv2.medianBlur(image, median_kernel_size)
#     dilate_kernel_size = random.randint(7, 12)
#     dilate_iterations = random.randint(4, 6)
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_kernel_size, dilate_kernel_size))
#     bled_image_final = cv2.dilate(bled_image_base, kernel, iterations=dilate_iterations)
#     stain_color = np.full_like(bled_image_final, [10, 40, 60], dtype=np.uint8)
#     bled_image_final = cv2.addWeighted(bled_image_final, 0.85, stain_color, 0.15, 0)
#     output_image = (bled_image_final * smooth_mask_3d + image * (1 - smooth_mask_3d)).astype(np.uint8)
#     return output_image