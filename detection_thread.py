from PyQt5.QtCore import QThread, pyqtSignal
import numpy as np
import cv2
import time
import os
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils.plotting import save_one_box
from ultralytics.utils.my_dataloader import ImageDataset, DataLoader
from ultralytics.reid_strong_baseline_master.modeling.baseline import Baseline
from torchvision import transforms
from database_manager import DatabaseManager
import torch
import torch.backends.cudnn as cudnn

class DetThread(QThread):
    # 自定义信号
    send_img = pyqtSignal(np.ndarray)
    send_raw = pyqtSignal(np.ndarray)
    send_statistic_img = pyqtSignal(np.ndarray)
    send_statistic_text = pyqtSignal(str)
    send_msg = pyqtSignal(str)
    send_percent = pyqtSignal(int)
    send_fps = pyqtSignal(str)
    send_high_similarity_frames = pyqtSignal(list)

    def __init__(self):
        super(DetThread, self).__init__()
        self.weights_yolo = ''
        self.weights_reid = ''
        self.source = '0'
        self.target = ''
        self.conf_thres_yolo = 0.6
        self.conf_thres_reid = 0.25
        self.iou_thres_yolo = 0.5
        self.jump_out = False
        self.is_continue = True
        self.percent_length = 1000
        self.save_fold = './result'
        self.db_manager = DatabaseManager()
        self.auto_detect = False
        self.best_yolo_thres = 0.2
        self.found_best_threshold = False
        self.last_similarities = []
        self.max_similarity_history = 5
        self.min_similarity = 0.89
        self.stable_count = 0
        self.required_stable_frames = 5
        self.last_bbox = None
        self.max_movement = 50
        self.position_history = []
        self.max_position_history = 5
        self.smooth_factor = 0.3
        self.high_similarity_frames = []
        self.current_frame = 0
        self.avg_high_similarity = 0.89
        self.red_box_count = 0
        self.red_box_threshold = 0.95
        self.green_box_threshold = 0.8
        self.orange_box_threshold = 0.89
        self.first_red_box_avg = 0.89

    def __del__(self):
        self.db_manager.close()

    def calculate_iou(self, bbox1, bbox2):
        """计算两个边界框的IoU"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
            
        intersection = (x2 - x1) * (y2 - y1)
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        return intersection / float(bbox1_area + bbox2_area - intersection)

    def calculate_center_distance(self, bbox1, bbox2):
        """计算两个边界框中心点的距离"""
        if bbox1 is None or bbox2 is None:
            return float('inf')
        center1 = ((bbox1[0] + bbox1[2])/2, (bbox1[1] + bbox1[3])/2)
        center2 = ((bbox2[0] + bbox2[2])/2, (bbox2[1] + bbox2[3])/2)
        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

    def init_kalman(self, bbox):
        """初始化卡尔曼滤波器"""
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
        self.kalman.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], np.float32) * 0.03
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        self.kalman.statePre = np.array([[center_x], [center_y], [0], [0]], np.float32)
        self.kalman.statePost = np.array([[center_x], [center_y], [0], [0]], np.float32)

    def predict_next_position(self):
        """使用卡尔曼滤波器预测下一帧位置"""
        if self.kalman is not None:
            prediction = self.kalman.predict()
            return prediction[0][0], prediction[1][0]
        return None, None

    def update_kalman(self, bbox):
        """更新卡尔曼滤波器状态"""
        if self.kalman is not None:
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            measurement = np.array([[center_x], [center_y]], np.float32)
            self.kalman.correct(measurement)

    def calculate_velocity(self, current_bbox, last_bbox):
        """计算目标运动速度"""
        if last_bbox is None:
            return [0, 0]
        dx = (current_bbox[0] + current_bbox[2])/2 - (last_bbox[0] + last_bbox[2])/2
        dy = (current_bbox[1] + current_bbox[3])/2 - (last_bbox[1] + last_bbox[3])/2
        return [dx, dy]

    def smooth_position(self, current_bbox):
        """平滑处理边界框位置"""
        if self.last_bbox is None:
            return current_bbox
        
        smoothed_bbox = [
            self.last_bbox[0] + self.smooth_factor * (current_bbox[0] - self.last_bbox[0]),
            self.last_bbox[1] + self.smooth_factor * (current_bbox[1] - self.last_bbox[1]),
            self.last_bbox[2] + self.smooth_factor * (current_bbox[2] - self.last_bbox[2]),
            self.last_bbox[3] + self.smooth_factor * (current_bbox[3] - self.last_bbox[3])
        ]
        return smoothed_bbox

    def run(self):
        try:
            count = 0
            org_video_path = self.source
            target_img = self.target
            
            print('-----------正在加载模型-------------')
            yolo_model = YOLO(self.weights_yolo)
            print("yolo模型加载完成 Model size: {:.5f}M".format(sum(p.numel() for p in yolo_model.parameters())/1000000.0))

            cudnn.benchmark = True

            reid_model = Baseline(751, 1, '', 'bnneck', 'after','resnet50', 'self')
            reid_model.load_param(self.weights_reid)
            print("reid模型加载完成 Model size: {:.5f}M".format(sum(p.numel() for p in reid_model.parameters())/1000000.0))
            
            if torch.cuda.is_available():
                reid_model = reid_model.cuda()
                yolo_model = yolo_model.cuda()
                reid_model.eval()

            print('-----------使用目标检测模型裁剪target图像------------')
            resultss = yolo_model.predict(target_img, name='test_crop', conf=self.conf_thres_yolo)

            if resultss[0].__len__() != 1:
                raise ValueError("target中没有目标。target规定为1个人")
    
            else:
                target_imgs=[]
                for _,b in enumerate(resultss[0].boxes):
                    crop_target_img = save_one_box(b.xyxy,
                                resultss[0].orig_img,
                                file=Path(''),
                                BGR=True,
                                save=False)
                    target_imgs.append([crop_target_img,0,0]) 
            
            print('-----------使用reid模型提取target图像特征-------------')
            image_transform = transforms.Compose([transforms.Resize((256,128)),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

            targer_img_dataloader = DataLoader(ImageDataset(target_imgs, transforms=image_transform), 
                                            batch_size=1, shuffle=False,num_workers=0,
                                            pin_memory=False, drop_last=False)

            with torch.no_grad():
                qf = []
                for _,(imgs, _, _) in enumerate(targer_img_dataloader):
                    imgs = imgs.cuda()
                    features = reid_model(imgs)
                    features = features.data.cpu()
                    qf.append(features)
                qf = torch.cat(qf, 0)

            if self.source == '0':
                cap = cv2.VideoCapture(int(org_video_path))
            elif self.source.split('/')[-1].split('.')[-1] == "mp4":
                cap = cv2.VideoCapture(org_video_path)
           
            if self.save_fold:
                os.makedirs(self.save_fold, exist_ok=True)
                pre_video_save_path = self.save_fold + f'/{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}.mp4'
                fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
                out_fps = cap.get(cv2.CAP_PROP_FPS)
                output_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                output_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                out = cv2.VideoWriter(pre_video_save_path, fourcc, out_fps, (output_width, output_height))
            
            ret, _ = cap.read()
            if not ret:
                raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）")
          
            while True:
                if self.jump_out:
                    self.send_high_similarity_frames.emit(self.high_similarity_frames)
                    cap.release()
                    self.send_percent.emit(0)
                    self.send_msg.emit('Stop')
                    if self.save_fold:
                        print("视频处理结果保存到 :" + pre_video_save_path)
                        out.release()
                    break

                if self.is_continue:
                    success, frame = cap.read()
                    if success:
                        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                        fps_start = time.time()
                        if self.source == '0':
                            frame_id = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                        else:
                            frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES)-1)

                        print('-----------正在处理第{}帧图像-----------'.format(frame_id))

                        time1 = time.time()
                        results = yolo_model.predict(frame,conf=self.conf_thres_yolo,iou=self.iou_thres_yolo,
                                                    name='yolo_predict')
        
                        time2 = time.time()
                        print(f" v8 耗时: {time2-time1} ")

                        if results[0].__len__() == 0:
                            self.send_raw.emit(frame)
                            self.send_img.emit(frame)
                            continue

                        self.send_raw.emit(frame)
                        crop_imgs=[]
                        for b in results[0].boxes:
                            crop_img = save_one_box(b.xyxy,
                                    results[0].orig_img,
                                    file=Path(''),
                                    BGR=True,
                                    save=False)
                            crop_imgs.append([crop_img,b.xyxy,frame_id])

                        crop_imgs_dataloader = DataLoader(ImageDataset(crop_imgs, transforms=image_transform), 
                                                batch_size=8, shuffle=False,num_workers=0,
                                                pin_memory=False, drop_last=False)
                        s = time.time()
                        with torch.no_grad():
                            gf = []
                            for _,(imgs, _, _) in enumerate(crop_imgs_dataloader):
                                imgs = imgs.cuda()
                                features = reid_model(imgs)
                                features = features.data.cpu()
                                gf.append(features)
                            gf = torch.cat(gf, 0)
                            e = time.time()
                            print(f" reid 耗时: {e-s} ")
                            
                            s = time.time()
                            qf = 1. * qf / (torch.norm(qf, 2, dim = -1, keepdim=True).expand_as(qf) + 1e-12)
                            gf = 1. * gf / (torch.norm(gf, 2, dim = -1, keepdim=True).expand_as(gf) + 1e-12)
                            m, n = qf.size(0), gf.size(0)
                            distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                                    torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
  
                            indices = np.argsort(distmat.addmm_(qf, gf.t(),beta=1,alpha=-2).numpy(), axis=1)
                            distance = distmat[0][indices[0][0]].cpu().numpy()
                            distance = 2 / (1 + np.exp(-distance)) - 1
                            pre_best_similarity = np.sqrt(1 - distance**2)

                            e = time.time()
                            print(f" dist 耗时: {e-s} ")

                            s = time.time()
                            if self.auto_detect and not self.found_best_threshold:
                                search_threshold = (0.89 + self.first_red_box_avg) / 2
                                if pre_best_similarity >= search_threshold:
                                    self.last_similarities.append(pre_best_similarity)
                                    if len(self.last_similarities) > self.max_similarity_history:
                                        self.last_similarities.pop(0)
                                    
                                    if len(self.last_similarities) >= self.required_stable_frames:
                                        avg_similarity = sum(self.last_similarities) / len(self.last_similarities)
                                        if avg_similarity >= search_threshold:
                                            self.stable_count += 1
                                            if self.stable_count >= self.required_stable_frames:
                                                self.best_yolo_thres = self.conf_thres_yolo
                                                self.conf_thres_reid = self.red_box_threshold
                                                self.found_best_threshold = True
                                                self.avg_high_similarity = avg_similarity
                                                self.red_box_count = 1
                                                self.first_red_box_avg = avg_similarity
                                                self.send_msg.emit(f'找到稳定的高相似度目标: YOLO={self.best_yolo_thres:.2f}, ReID={self.red_box_threshold:.2f}, 平均相似度={avg_similarity:.3f}')
                                    else:
                                        self.stable_count = 0
                                else:
                                    self.conf_thres_reid += 0.01
                                    if self.conf_thres_reid > self.red_box_threshold:
                                        self.conf_thres_reid = search_threshold
                                    self.last_similarities = []
                                    self.stable_count = 0
                            elif self.auto_detect and self.found_best_threshold:
                                search_threshold = (0.89 + self.first_red_box_avg) / 2
                                if pre_best_similarity >= search_threshold:
                                    self.conf_thres_yolo = self.best_yolo_thres
                                    if pre_best_similarity >= self.red_box_threshold:
                                        self.conf_thres_reid = self.red_box_threshold
                                        self.red_box_count += 1
                                        self.avg_high_similarity = (self.avg_high_similarity * (self.red_box_count - 1) + pre_best_similarity) / self.red_box_count
                                    else:
                                        self.conf_thres_reid = max(search_threshold, pre_best_similarity - 0.01)
                                else:
                                    self.found_best_threshold = False
                                    self.last_similarities = []
                                    self.stable_count = 0
                                    self.conf_thres_reid = search_threshold
                                    self.send_msg.emit(f'目标丢失，重新搜索中... 使用阈值{(0.89 + self.first_red_box_avg) / 2:.3f}')

                            if pre_best_similarity >= self.red_box_threshold:
                                color = (0, 0, 255)
                            elif pre_best_similarity >= self.orange_box_threshold:
                                color = (0, 165, 255)
                            elif pre_best_similarity >= self.green_box_threshold:
                                color = (0, 255, 0)
                            else:
                                color = None

                            if color is not None:
                                x1, y1, x2, y2 = crop_imgs[indices[0][0]][1][0]
                                cv2.putText(results[0].orig_img, f"{pre_best_similarity:.2f}", 
                                          (int(x2), int(y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                                cv2.rectangle(results[0].orig_img, (int(x1), int(y1)), 
                                            (int(x2), int(y2)), color, 2)
                                
                                target_id = os.path.basename(self.target)
                                frame_id = str(crop_imgs[indices[0][0]][2])
                                image_path = os.path.join(self.save_fold, f'matched_in_{frame_id}.jpg') if self.save_fold else None
                                
                                detection_box = (int(x1), int(y1), int(x2-x1), int(y2-y1))
                                
                                source_type = '摄像头' if self.source == '0' else '视频文件' if self.source.endswith(('.mp4', '.avi', '.mkv')) else 'RTSP流'
                                
                                try:
                                    self.db_manager.insert_detection(
                                        target_id=target_id,
                                        frame_id=frame_id,
                                        similarity=float(pre_best_similarity),
                                        image_path=image_path,
                                        detection_box=detection_box,
                                        source_type=source_type,
                                        source_path=self.source
                                    )
                                except Exception as e:
                                    print(f"保存检测结果到数据库失败: {e}")
                    
                            if self.save_fold:
                                out.write(results[0].orig_img)

                            e = time.time()
                            fps_stop = time.time()
                            print(f" vis 耗时: {e-s} ")
                            fps = int(1/(fps_stop-fps_start))
                            self.send_fps.emit('fps：'+str(fps))
                            self.send_img.emit(results[0].orig_img)

                            count += 1
                            percent = int(count/cap.get(cv2.CAP_PROP_FRAME_COUNT)*self.percent_length)
                            self.send_percent.emit(percent)

                            if  pre_best_similarity >= self.conf_thres_reid:
                                best_match_img = crop_imgs[indices[0][0]][0]
                                self.send_statistic_img.emit(best_match_img) 
                                self.send_statistic_text.emit(str(crop_imgs[indices[0][0]][2]))

                                if self.found_best_threshold and pre_best_similarity >= 0.9 and self.conf_thres_reid == 0.9:
                                    self.high_similarity_frames.append(current_frame)
                                    self.high_similarity_frames = sorted(list(set(self.high_similarity_frames)))

                    else:
                        self.send_high_similarity_frames.emit(self.high_similarity_frames)
                        break

        except Exception as e:
            self.send_msg.emit('%s' % e)
            print('%s' % e) 