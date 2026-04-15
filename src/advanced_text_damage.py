import os
import datetime
import numpy as np
from PIL import Image
from opensimplex import OpenSimplex # 你可能需要安装这个库：pip install opensimplex
import random
import cv2 # 用于图像处理，特别是转换为OpenCV格式

# --- 核心生成函数 (回归到最成功的版本) ---

def fractal_noise(simplex, x, y, octaves, persistence, lacunarity):
    """生成分形噪声，用于创建基础纹理。"""
    frequency = 1.0
    amplitude = 1.0
    total = 0.0
    max_amplitude = 0.0
    for _ in range(octaves):
        total += simplex.noise2(x * frequency, y * frequency) * amplitude
        max_amplitude += amplitude
        amplitude *= persistence
        frequency *= lacunarity
    return total / max_amplitude

def create_sprawling_gradient(width, height, simplex, distortion_strength, distortion_scale, power):
    """创建被噪声扭曲的、自身就带有触手和分支的渐变。"""
    center_x, center_y = width / 2, height / 2
    y_grid, x_grid = np.ogrid[:height, :width]
    x_dist = np.zeros((height, width))
    y_dist = np.zeros((height, width))
    # OpenSimplex噪声的输入通常是浮点数
    for i in range(height):
        for j in range(width):
            x_dist[i, j] = simplex.noise2(i / distortion_scale, j / distortion_scale)
            y_dist[i, j] = simplex.noise2((i + 500) / distortion_scale, (j + 500) / distortion_scale)
    
    distorted_x = (x_grid - center_x) + x_dist * distortion_strength
    distorted_y = (y_grid - center_y) + y_dist * distortion_strength
    
    dist_from_center = np.sqrt(distorted_x**2 + distorted_y**2)
    max_dist = np.sqrt((width/2)**2 + (height/2)**2)
    normalized_dist = dist_from_center / max_dist
    
    gradient = 1.0 - np.power(normalized_dist, power)
    gradient[gradient < 0] = 0
    return gradient

def generate_damage_mask_pil(width=512, height=512, seed=None, recipe='classic'):
    """
    主生成函数，现在支持多种“配方”(recipe)。
    返回 PIL Image 格式的蒙版。
    """
    if seed is None:
        seed = random.randint(0, 10000)
        
    # --- 参数配方区 ---
    # 您可以在这里定义或修改不同的风格
    if recipe == 'classic':
        # 我们之前最成功的版本，平衡了细节和延伸
        p = {
            'shape_scale': 150.0, 'shape_octaves': 6, 'threshold': 0.5,
            'dist_strength': 130.0, 'dist_scale': 250.0, 'power': 1.5
        }
    elif recipe == 'compact_blot':
        # 更接近第二版原始效果，紧凑，细节丰富，延伸较少
        p = {
            'shape_scale': 120.0, 'shape_octaves': 7, 'threshold': 0.55,
            'dist_strength': 80.0, 'dist_scale': 300.0, 'power': 2.0
        }
    elif recipe == 'aggressive_spread':
        # 更加奔放，延伸出更多细长的触手
        p = {
            'shape_scale': 180.0, 'shape_octaves': 5, 'threshold': 0.45,
            'dist_strength': 200.0, 'dist_scale': 200.0, 'power': 1.2
        }
    else: # 默认使用经典配方
        p = {
            'shape_scale': 150.0, 'shape_octaves': 6, 'threshold': 0.5,
            'dist_strength': 130.0, 'dist_scale': 250.0, 'power': 1.5
        }

    simplex_shape = OpenSimplex(seed=seed)
    simplex_gradient = OpenSimplex(seed=seed + 1)
    
    world = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            world[i][j] = fractal_noise(simplex_shape, i / p['shape_scale'], j / p['shape_scale'], p['shape_octaves'], 0.5, 2.0)
    world = (world + 1) / 2 # 归一化到 [0, 1]
    
    gradient = create_sprawling_gradient(width, height, simplex_gradient,
                                         p['dist_strength'], p['dist_scale'], p['power'])
    
    final_world = world * gradient
    mask_data = np.where(final_world > p['threshold'], 0, 255).astype(np.uint8) # 0 for damage, 255 for no damage
    mask_image = Image.fromarray(mask_data, 'L')
    return mask_image

def apply_advanced_text_damage(image, text_mask, background_patch, damage_recipe='classic', damage_intensity=0.15):
    """
    将高级文字破坏蒙版应用于图像。
    这个函数会生成一个不规则的“墨迹破坏”蒙版，然后用背景替换掉文字区域被蒙版覆盖的部分。

    Args:
        image (np.array): 当前的退化图像 (BGR)。
        text_mask (np.array): 文本的二值蒙版 (单通道，255为文本，0为背景)。
        background_patch (np.array): 背景图像块 (BGR)。
        damage_recipe (str): 损伤生成配方 ('classic', 'compact_blot', 'aggressive_spread')。
        damage_intensity (float): 破坏蒙版叠加到文字蒙版时的强度 (0到1之间)。

    Returns:
        np.array: 应用了高级文字破坏的图像。
    """
    output_image = image.copy()
    h, w = image.shape[:2]

    # 生成一个墨迹破坏蒙版 (PIL Image)
    pil_damage_mask = generate_damage_mask_pil(width=w, height=h, recipe=damage_recipe)
    
    # 转换为 OpenCV 格式的 numpy 数组
    damage_mask = np.array(pil_damage_mask)

    # damage_mask 是 0 (破坏) 和 255 (非破坏)，我们希望它反过来表示破坏区域
    # 或者，我们直接将其用作一个“擦除”的强度蒙版
    # 将其归一化到 0-1
    damage_mask_float = damage_mask.astype(np.float32) / 255.0
    
    # 根据 damage_intensity 调整破坏蒙版，并与文本蒙版结合
    # 这里的逻辑是：文字蒙版是255，表示文字存在；damage_mask_float是0表示破坏
    # 我们希望在文字存在且damage_mask_float为0的地方进行替换
    
    # 将 text_mask 归一化到 0-1 (0为背景，1为文字)
    text_mask_float = text_mask.astype(np.float32) / 255.0

    # 组合蒙版：只有在文本区域且破坏蒙版强度足够的地方才进行破坏
    # 如果 damage_mask_float 越接近 0 (即原始 damage_mask 越接近 255)，破坏越少
    # 如果 damage_mask_float 越接近 1 (即原始 damage_mask 越接近 0)，破坏越多
    # 我们需要的是 damage_mask 中 0 的部分来表示破坏
    # 因此，我们取 (1 - damage_mask_float) 来表示“破坏强度”
    final_damage_area = text_mask_float * (1.0 - damage_mask_float)
    
    # 使用随机阈值来决定最终的破坏区域，引入随机性
    # 只有当 `final_damage_area` 达到一定强度才认为是一个破坏点
    random_threshold = random.uniform(0.01, damage_intensity) # 引入随机强度
    final_damage_binary_mask = (final_damage_area > random_threshold).astype(np.uint8) * 255

    # 找到所有需要破坏的像素坐标
    damage_coords = np.argwhere(final_damage_binary_mask > 0)

    # 用背景像素替换
    if len(damage_coords) > 0:
        rows, cols = damage_coords[:, 0], damage_coords[:, 1]
        output_image[rows, cols] = background_patch[rows, cols]
        
    return output_image


if __name__ == "__main__":
    # 示例用法，如果你直接运行这个文件
    # 需要一个dummy image和background
    dummy_image = np.full((512, 512, 3), 128, dtype=np.uint8) # 灰色图像
    dummy_background = np.full((512, 512, 3), 200, dtype=np.uint8) # 浅灰色背景
    
    # 绘制一个简单的文本作为mask
    text_color = (0, 0, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(dummy_image, "Hello Manchu", (50, 250), font, 2, text_color, 4, cv2.LINE_AA)
    
    gray_image = cv2.cvtColor(dummy_image, cv2.COLOR_BGR2GRAY)
    _, dummy_text_mask = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY_INV) # 假设文字较深

    print("Generating advanced text damage example...")
    degraded_img = apply_advanced_text_damage(dummy_image.copy(), dummy_text_mask, dummy_background.copy(), damage_recipe='classic', damage_intensity=0.3)
    
    output_folder = "advanced_text_damage_test"
    os.makedirs(output_folder, exist_ok=True)
    cv2.imwrite(os.path.join(output_folder, "original_with_text.png"), dummy_image)
    cv2.imwrite(os.path.join(output_folder, "text_mask.png"), dummy_text_mask)
    cv2.imwrite(os.path.join(output_folder, "degraded_advanced_text_damage.png"), degraded_img)
    print(f"Example saved to {output_folder}")