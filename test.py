import os
import glob
from mmdet.apis import init_detector, inference_detector
from tqdm import tqdm
import mmcv

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']
def is_img_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

# 指定模型的配置文件和 checkpoint 文件路径
config_file = './work_dirs/cascade_mask_rcnn_hrnetv2p_w32_20e_v2/cascade_mask_rcnn_hrnetv2p_w32_20e_v2.py'
checkpoint_file = './work_dirs/cascade_mask_rcnn_hrnetv2p_w32_20e_v2/epoch_24.pth'

# 根据配置文件和 checkpoint 文件构建模型
model = init_detector(config_file, checkpoint_file, device='cuda:0')

test_dir = '../datasets/pubtest/imgs/'
results_dir = './work_dirs/cascade_mask_rcnn_hrnetv2p_w32_20e_v2/results_epoch24/'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

images_dir = [os.path.join(test_dir, x) for x in os.listdir(test_dir) if is_img_file(x)]
for img_dir in tqdm(images_dir):
    result = inference_detector(model, img_dir)
    model.show_result(img_dir, result, out_file=os.path.join(results_dir, os.path.basename(img_dir)))

# print(images)
# # 测试单张图片并展示结果
# img = 'test.jpg'  # 或者 img = mmcv.imread(img)，这样图片仅会被读一次
# result = inference_detector(model, img)
# # 在一个新的窗口中将结果可视化
# model.show_result(img, result)
# # 或者将可视化结果保存为图片
# model.show_result(img, result, out_file='result.jpg')