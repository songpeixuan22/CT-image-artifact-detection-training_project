from torchvision import transforms
from PIL import Image
import numpy as np

def label_pos(dataset):
    sum_x, sum_y, num = 0,0,0
    for images, labels in dataset:
        # 假设白色部分的像素值为 255
        labels = np.squeeze(np.array(labels))
        white_pixels = np.squeeze(np.where(labels >= 200))
        print(white_pixels)
        if white_pixels != []:
            sum_x += white_pixels[1].sum().item()
            sum_y += white_pixels[0].sum().item()
            num += len(white_pixels[0])

    if num > 0:
        avg_x = sum_x / num
        avg_y = sum_y / num
        print(f"平均坐标值: ({avg_x}, {avg_y})")
        return avg_x, avg_y
    else:
        print("没有找到白色部分")
        return None, None