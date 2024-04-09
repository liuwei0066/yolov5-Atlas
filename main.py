# 导入代码依赖
import cv2
import numpy as np
import ipywidgets as widgets
from IPython.display import display
import torch
from skvideo.io import vreader, FFmpegWriter
import IPython.display
from ais_bench.infer.interface import InferSession

from det_utils import letterbox, scale_coords, nms

import json
import os
import glob
import argparse

def ensure_output_dir(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)



def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def coco80_to_coco91_class():
    # converts 80-index (val2014/val2017) to 91-index (paper)
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
         35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
         64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    return x


def xyxy2xywh_single(bbox):
    # Convert a single box from [x1, y1, x2, y2] to [x_center, y_center, width, height]
    x_center = (bbox[0] + bbox[2]) / 2
    y_center = (bbox[1] + bbox[3]) / 2
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    return np.array([x_center, y_center, width, height])



def preprocess_image(image, cfg, bgr2rgb=True):
    """图片预处理"""
    img, scale_ratio, pad_size = letterbox(image, new_shape=cfg['input_shape'])
    if bgr2rgb:
        img = img[:, :, ::-1]
    img = img.transpose(2, 0, 1)  # HWC2CHW
    img = np.ascontiguousarray(img, dtype=np.float32)/255.0
    #img = np.ascontiguousarray(img, dtype=np.float32)
    return img, scale_ratio, pad_size


def draw_bbox(bbox, img0, color, wt, names):
    """在图片上画预测框"""
    det_result_str = ''
    for idx, class_id in enumerate(bbox[:, 5]):
        if float(bbox[idx][4] < float(0.05)):
            continue
        img0 = cv2.rectangle(img0, (int(bbox[idx][0]), int(bbox[idx][1])), (int(bbox[idx][2]), int(bbox[idx][3])),
                             color, wt)
        img0 = cv2.putText(img0, str(idx) + ' ' + names[int(class_id)], (int(bbox[idx][0]), int(bbox[idx][1] + 16)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        img0 = cv2.putText(img0, '{:.4f}'.format(bbox[idx][4]), (int(bbox[idx][0]), int(bbox[idx][1] + 32)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        det_result_str += '{} {} {} {} {} {}\n'.format(
            names[bbox[idx][5]], str(bbox[idx][4]), bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3])
    return img0


def get_labels_from_txt(path):
    """从txt文件获取图片标签"""
    labels_dict = dict()
    with open(path) as f:
        for cat_id, label in enumerate(f.readlines()):
            labels_dict[cat_id] = label.strip()
    return labels_dict


def draw_prediction(pred, image, labels,output_path):
    """在图片上画出预测框并进行可视化展示"""
    imgbox = widgets.Image(format='jpg', height=720, width=1280)
    img_dw = draw_bbox(pred, image, (0, 255, 0), 2, labels)
    imgbox.value = cv2.imencode('.jpg', img_dw)[1].tobytes()
    #display(imgbox)
    cv2.imwrite(output_path, img_dw)



def infer_map(img_path, model, class_names, cfg, image_id, all_detections):
    """图片推理"""
    # 图片载入
    image = cv2.imread(img_path)
    # 数据预处理
    img, scale_ratio, pad_size = preprocess_image(image, cfg)
    # 模型推理
    output = model.infer([img])[0]

    output = torch.tensor(output)
    # 非极大值抑制后处理
    boxout = nms(output, conf_thres=cfg["conf_thres"], iou_thres=cfg["iou_thres"])
    pred_all = boxout[0].numpy()
#    print(pred_all)
    # 预测坐标转换
    scale_coords(cfg['input_shape'], pred_all[:, :4], image.shape, ratio_pad=(scale_ratio, pad_size))
#    print(pred_all)

    for pred in pred_all:
        x1, y1, x2, y2, confidence, class_id = pred
        # 确保坐标和置信度是Python原生的float类型
        bbox = [float(x1), float(y1), float(x2), float(y2)]
        bbox = xyxy2xywh_single(bbox)
        bbox[0] -= bbox[2] / 2
        bbox[1] -= bbox[3] / 2
        # Round the bbox values to 3 decimal places and score to 5 decimal places
        bbox = [round(coord, 3) for coord in bbox]
        confidence = round(float(confidence), 5)
        all_detections.append({
            "image_id": image_id,  # 你需要有一个图片ID
            "category_id": coco80_to_coco91_class()[int(class_id)],
            "bbox": bbox,  # COCO 格式是 [x, y, width, height]
            "score": confidence
        })


def infer_image(img_path, model, class_names, cfg, image_id, output_dir):
    """图片推理并保存结果"""
    # 图片载入
    image = cv2.imread(img_path)
    # 数据预处理
    img, scale_ratio, pad_size = preprocess_image(image, cfg)
    # 模型推理
    output = model.infer([img])[0]

    output = torch.tensor(output)
    # 非极大值抑制后处理
    boxout = nms(output, conf_thres=cfg["conf_thres"], iou_thres=cfg["iou_thres"])
    pred_all = boxout[0].numpy()
    # 预测坐标转换
    scale_coords(cfg['input_shape'], pred_all[:, :4], image.shape, ratio_pad=(scale_ratio, pad_size))

    # 确保输出目录存在
    ensure_output_dir(output_dir)
    output_file_path = os.path.join(output_dir, "{}.jpg".format(image_id))
    
    # 图片预测结果可视化
    draw_prediction(pred_all, image, class_names, output_file_path)


def save_detections_to_file(detections, file_path):
    with open(file_path, 'w') as file:
        json.dump(detections, file)

def infer_frame_with_vis(image, model, labels_dict, cfg, bgr2rgb=True):
    # 数据预处理
    img, scale_ratio, pad_size = preprocess_image(image, cfg, bgr2rgb)
    # 模型推理
    output = model.infer([img])[0]

    output = torch.tensor(output)
    # 非极大值抑制后处理
    boxout = nms(output, conf_thres=cfg["conf_thres"], iou_thres=cfg["iou_thres"])
    pred_all = boxout[0].numpy()
    # 预测坐标转换
    scale_coords(cfg['input_shape'], pred_all[:, :4], image.shape, ratio_pad=(scale_ratio, pad_size))
    # 图片预测结果可视化
    img_vis = draw_bbox(pred_all, image, (0, 255, 0), 2, labels_dict)
    return img_vis


def img2bytes(image):
    """将图片转换为字节码"""
    return bytes(cv2.imencode('.jpg', image)[1])


# def infer_video(video_path, model, labels_dict, cfg):
#     """视频推理"""
#     image_widget = widgets.Image(format='jpeg', width=800, height=600)
#     display(image_widget)

#     # 读入视频
#     cap = cv2.VideoCapture(video_path)
#     while True:
#         ret, img_frame = cap.read()
#         if not ret:
#             break
#         # 对视频帧进行推理
#         image_pred = infer_frame_with_vis(img_frame, model, labels_dict, cfg, bgr2rgb=True)
#         image_widget.value = img2bytes(image_pred)

# def infer_video(video_path, model, labels_dict, cfg, output_video_path):
#     """视频推理并保存结果到文件"""
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print(f"Error opening video file: {video_path}")
#         return
    
#     # 获取视频属性
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = cap.get(cv2.CAP_PROP_FPS)
    
#     # 创建视频写入器
#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
#     while True:
#         ret, img_frame = cap.read()
#         if not ret:
#             break
#         # 对视频帧进行推理
#         image_pred = infer_frame_with_vis(img_frame, model, labels_dict, cfg, bgr2rgb=True)
#         # 写入推理结果到输出视频
#         out.write(image_pred)
    
#     # 释放资源
#     cap.release()
#     out.release()
#     #cv2.destroyAllWindows()

def infer_video(video_path, model, labels_dict, cfg, output_video_path):
    """视频推理并保存结果到文件"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return
    
    # 获取视频属性
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # 创建视频写入器，使用MP4格式
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 或者 'avc1'
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    while True:
        ret, img_frame = cap.read()
        if not ret:
            break
        # 对视频帧进行推理
        image_pred = infer_frame_with_vis(img_frame, model, labels_dict, cfg, bgr2rgb=True)
        # 写入推理结果到输出视频
        out.write(image_pred)
    
    # 释放资源
    cap.release()
    out.release()



def infer_camera(model, labels_dict, cfg):
    """外设摄像头实时推理"""
    def find_camera_index():
        max_index_to_check = 10  # Maximum index to check for camera

        for index in range(max_index_to_check):
            cap = cv2.VideoCapture(index)
            if cap.read()[0]:
                cap.release()
                return index

        # If no camera is found
        raise ValueError("No camera found.")

    # 获取摄像头
    camera_index = find_camera_index()
    cap = cv2.VideoCapture(camera_index)
    # 初始化可视化对象
    image_widget = widgets.Image(format='jpeg', width=1280, height=720)
    display(image_widget)
    while True:
        # 对摄像头每一帧进行推理和可视化
        _, img_frame = cap.read()
        image_pred = infer_frame_with_vis(img_frame, model, labels_dict, cfg)
        image_widget.value = img2bytes(image_pred)


def parse_args():
    parser = argparse.ArgumentParser(description='YOLO Inference Script')
    parser.add_argument('--infer-mode', type=str, default='image', choices=['image', 'video', 'camera','map'], help='Inference mode: image, video, or camera')
    parser.add_argument('--images-folder', type=str, default='val2017', help='Folder containing images for inference')
    parser.add_argument('--video-path', type=str, default='racing.mp4', help='Path to the video file')
    parser.add_argument('--model-path', type=str, default='yolov5s-det.om', help='Path to the model file')
    parser.add_argument('--label-path', type=str, default='./coco_names.txt', help='Path to the label file')
    parser.add_argument('--output-json', type=str, default='./json/detections.json', help='Output path for JSON with detections')
    parser.add_argument('--output-dir', type=str, default='./out', help='Directory for saving output images')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = {
        'conf_thres': 0.0005,  # 模型置信度阈值
        'iou_thres': 0.5,  # IOU阈值
        'input_shape': [640, 640],  # 模型输入尺寸
    }
    
    # 初始化推理模型
    model = InferSession(0, args.model_path)
    labels_dict = get_labels_from_txt(args.label_path)
    if args.infer_mode == 'image':
        count = 0
        image_paths = []
        image_formats = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        for format in image_formats:
            image_paths.extend(glob.glob(os.path.join(args.images_folder, format)))
        output_dir = args.output_dir
        for img_path in image_paths:
            image_id = os.path.splitext(os.path.basename(img_path))[0]
            image_id = int(image_id)
            infer_image(img_path, model, labels_dict, cfg, image_id, output_dir)
            count += 1
            print("已经推理：",count,"张图片")

    elif args.infer_mode == 'camera':
        infer_camera(model, labels_dict, cfg)
    elif args.infer_mode == 'video':

        # 使用函数并指定输出视频路径
        infer_video(args.video_path, model, labels_dict, cfg, "output_video.mp4")
        #infer_video(args.video_path, model, labels_dict, cfg)
    elif args.infer_mode == 'map':
        all_detections = []
        image_paths = []
        image_formats = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        count = 0
        for format in image_formats:
            image_paths.extend(glob.glob(os.path.join(args.images_folder, format)))
        for img_path in image_paths:
            image_id = os.path.splitext(os.path.basename(img_path))[0]
            image_id = int(image_id)
            infer_map(img_path, model, labels_dict, cfg, image_id, all_detections)
            count += 1
            print("已经推理：",count,"张图片")
        # 确保输出 JSON 文件的目录存在
        json_output_dir = os.path.dirname(args.output_json)
        ensure_output_dir(json_output_dir)
        save_detections_to_file(all_detections, args.output_json)

if __name__ == '__main__':
    main()


