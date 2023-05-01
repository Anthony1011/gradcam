# GradCAM and GradCAM++ based on yolov7 model use pytorch framework
import cv2
import os
import numpy as np
import torch
import argparse
import time
from models.gradcam import YOLOV7GradCAM, YOLOV7GradCAMPP
from models.yolov7_object_detector import YOLOV7TorchObjectDetector
from PIL import Image

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model-path', type=str, default="weights/yolov7.pt", help='Path to the model')
parser.add_argument('--image-path', type=str, default='path/to/your/image.jpg', help='Path to the input image')
parser.add_argument('--output-dir', type=str, default='outputs', help='output dir')
parser.add_argument('--img-size', type=int, default=640, help="input image size")
parser.add_argument('--target-layer', type=str, default=['102_act', '103_act', '104_act'], nargs='+',
                    help='The layer hierarchical address to which gradcam will applied,'
                         ' the names should be separated by underline')
parser.add_argument('--method', type=str, default='gradcam', help='gradcam method: gradcam, gradcampp')
parser.add_argument('--device', type=str, default='cpu', help='cuda or cpu')

args = parser.parse_args()

def preprocess_image(image_path):
    image = Image.open(image_path) # 讀取圖像
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # 將圖像從 BGR 轉換為 RGB。
    image = cv2.resize(image, (640, 640)) # 將圖像調整為模型的輸入尺寸。
    image = image / 255.0 # 將像素值範圍從 [0, 255] 轉換為 [0, 1]。
    image = image.transpose(2, 0, 1)  # 將圖像從 (H, W, C) 轉換為 (C, H, W)。
    image = torch.from_numpy(image).float() # 將圖像的數據類型從 uint8 轉換為 float32。
    image = image.unsqueeze(0) # 在最前面添加一個 batch dimension。
    return image

def postprocess_image(image):
    image = image.squeeze(0) # remove batch dimension
    image = image.transpose(1, 2, 0) # 將張量從 (C, H, W) 轉換為 (H, W, C)。
    image = image * 255.0 # 將像素值範圍從 [0, 1] 轉換為 [0, 255]。
    image = image.astype(np.uint8) # 將像素值的數據類型轉換為 np.uint8。
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # 將圖像從 RGB 轉換為 BGR。
    return image

def postprocess_cam(cam):
    cam = cam - np.min(cam) # 將張量的最小值移動到 0。
    cam = cam / np.max(cam) # 將張量的最大值移動到 1。
    cam = np.uint8(cam * 255) # 將張量的數據類型轉換為 np.uint8。
    return cam

def grad_cam(image_path, model, target_layer): # gradcam and gradcam++ function
    image = preprocess_image(image_path) # drop images to preprocess image
    #因為target_layer有三層,這裡把每一層分別丟進去做一遍
    for target_layer in target_layer:
        if args.method == 'gradcam': # chose gradcam
            grad_cam = YOLOV7GradCAM(model=model, layer_name=target_layer,img_size=(args.img_size, args.img_size)) # gradcam = YOLOV7GradCAM方法
            mask, _ = grad_cam(image) # cam mask
            heatmap, result = visualize_cam(mask, image) # make heatmap
        elif args.method == 'gradcampp': # chose gradcam++
            grad_cam_pp = YOLOV7GradCAMPP(model=model, layer_name=target_layer,img_size=(args.img_size, args.img_size)) # gradcam++ = YOLOV7GradCAMPP方法
            mask, _ = grad_cam_pp(image) # cam mask
            heatmap, result = visualize_cam(mask, image) # make heatmap
        else:
            raise NotImplementedError # raise error
        return heatmap, result

def visualize_cam(mask, img): # 將 cam mask 與原圖結合
    heatmap = (255 * mask.squeeze()).type(torch.uint8).cpu().numpy() # 將張量轉換為 numpy 格式
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)
    b, g, r = heatmap.split(1)
    heatmap = torch.cat([r, g, b])
    result = heatmap + img.cpu()
    result = result.div(result.max()).squeeze()
    return heatmap, result

def main(img_path, item):
    ## load model ##
    # 讀取指定模型(model_path)的權重
    model = torch.load(args.model_path, map_location=args.device)['model'].float()
    # 重建模型並把模型配置到指定裝置上
    model.to(args.device).eval()
    # gradcam and gradcam++ function, heatmap 是熱力圖本身. result 是熱力圖與原圖結合的圖片
    heatmap, result = grad_cam(img_path, model, args.target_layer)
    # 後處理熱力圖
    heatmap = postprocess_cam(heatmap)
    # 後處理結合圖
    result = postprocess_image(result)
    if not os.path.exists(args.output_dir):  # if output dir not exist
        os.makedirs(args.output_dir)  # make output dir
    print(f'[INFO] Saving result into {args.output_dir}')
    cv2.imwrite(os.path.join(args.output_dir, f'heatmap_{item}'), heatmap)
    cv2.imwrite(os.path.join(args.output_dir, f'result_{item}'), result)

if __name__ == '__main__':
    time_start = time.time() # 計時開始
    if os.path.isdir(args.image_path):  # 判定是否為資料夾
        img_list = os.listdir(args.image_path)  # 讀取文件夾中的圖片名
        print(img_list)  # 印出所有圖片名
        for item in img_list:
            main(os.path.join(args.image_path, item), item)  # 依序把每張圖組合成路徑後丟進去產生熱力圖
        else:
            item = args.image_path.split('/')[-1]  # 取得圖片名
            main(args.image_path, item)  # 圖片路徑為單個圖片
    time_end = time.time() # 計時結束
    print('time cost', time_end - time_start, 's') # 印出花費時間