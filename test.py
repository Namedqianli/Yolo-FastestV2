import os
import cv2
import time
import argparse
import torch
import model.detector
import utils.utils

def process_frame(frame, model, cfg, device, label_names):
    """
    对单帧图像进行推理、后处理及绘制
    """
    h_ori, w_ori, _ = frame.shape
    
    # 数据预处理
    res_img = cv2.resize(frame, (cfg["width"], cfg["height"]), interpolation=cv2.INTER_LINEAR)
    img = res_img.reshape(1, cfg["height"], cfg["width"], 3)
    img = torch.from_numpy(img.transpose(0, 3, 1, 2))
    img = img.to(device).float() / 255.0

    # 模型推理
    start = time.perf_counter()
    with torch.no_grad(): # 推理模式关闭梯度计算
        preds = model(img)
    end = time.perf_counter()
    
    inf_time = (end - start) * 1000.

    # 特征图后处理
    output = utils.utils.handel_preds(preds, cfg, device)
    output_boxes = utils.utils.non_max_suppression(output, conf_thres=0.3, iou_thres=0.4)

    # 绘制预测框
    scale_h, scale_w = h_ori / cfg["height"], w_ori / cfg["width"]
    for box in output_boxes[0]:
        box = box.tolist()
        obj_score = box[4]
        category = label_names[int(box[5])]

        x1, y1 = int(box[0] * scale_w), int(box[1] * scale_h)
        x2, y2 = int(box[2] * scale_w), int(box[3] * scale_h)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(frame, f'{category} {obj_score:.2f}', (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return frame, inf_time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='', help='Specify training profile *.data')
    parser.add_argument('--weights', type=str, default='', help='The path of the .pth model')
    parser.add_argument('--img', type=str, default='', help='The path of test image (optional)')
    parser.add_argument('--cam', type=int, default=None, help='Camera index (e.g., 0)')

    opt = parser.parse_args()
    cfg = utils.utils.load_datafile(opt.data)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.detector.Detector(cfg["classes"], cfg["anchor_num"], True, backbone=cfg["backbone"]).to(device)
    model.load_state_dict(torch.load(opt.weights, map_location=device))
    model.eval()

    # 加载标签
    LABEL_NAMES = []
    with open(cfg["names"], 'r') as f:
        LABEL_NAMES = [line.strip() for line in f.readlines()]

    # 情况 1: 静态图片识别
    if opt.img and os.path.exists(opt.img):
        print(f"Processing image: {opt.img}")
        ori_img = cv2.imread(opt.img)
        res_frame, _ = process_frame(ori_img, model, cfg, device, LABEL_NAMES)
        cv2.imwrite("test_result.png", res_frame)
        cv2.imshow("Detection Result", res_frame)
        cv2.waitKey(0)

    # 情况 2: 摄像头实时识别
    if opt.cam is not None:
        print(f"Starting Camera: {opt.cam}")
        cap = cv2.VideoCapture(opt.cam)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame, dt = process_frame(frame, model, cfg, device, LABEL_NAMES)
            
            # 显示 FPS/耗时
            cv2.putText(processed_frame, f"Time: {dt:.1f}ms", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            cv2.imshow("Real-time Detection (Press 'Q' to quit)", processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
    
    cv2.destroyAllWindows()