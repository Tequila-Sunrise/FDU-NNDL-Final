import os
from PIL import Image

shot = 1  # 1, 3, 5

dir_from = f'Oracle/sketched/oracle_{shot}shot'
dir_to = f'Oracle/oracle_{shot}shot'
os.makedirs(dir_to, exist_ok=True)

with open('Oracle/oracle_fs/seq/char_to_idx.txt', 'r', encoding='utf-8') as f:
    idx_to_cls = f.read()

imgs = os.listdir(dir_from)
for img in imgs:
    if img.endswith('.png'):  # sketched_{idx}_{proba}.png
        img_names = img.split('_')
        idx = int(img_names[1])
        proba = img_names[2].split('.')[0]

        folder_name = idx_to_cls[idx // shot]
        os.makedirs(os.path.join(dir_to, folder_name), exist_ok=True)

        img_path = os.path.join(dir_from, img)
        img = Image.open(img_path)
        img.save(os.path.join(dir_to, folder_name, f'{idx}_{proba}.png'))
