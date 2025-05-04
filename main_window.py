from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMenu, QAction, QLabel, QPushButton, QSlider, QProgressBar
from main_ui.main_ui import Ui_mainWindow
from PyQt5.QtCore import Qt, QPoint, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QPainter, QIcon
from utils.CustomMessageBox import MessageBox
from utils.capnums import Camera
from dialog_ui.rtsp_win import Window
import sys
import os
import json
import cv2
from detection_thread import DetThread

class MainWindow(QMainWindow, Ui_mainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.m_flag = False
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint
                           | Qt.WindowSystemMenuHint | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint)

        self.load_setting()
        self.blind_slots()

    def blind_slots(self):
        self.minButton.clicked.connect(self.showMinimized)
        self.maxButton.clicked.connect(self.max_or_restore)
        self.maxButton.animateClick(10)
        self.closeButton.clicked.connect(self.close)

        self.targetbox.clicked.connect(self.open_target)
        self.fileButton.clicked.connect(self.chose_file)
        self.cameraButton.clicked.connect(self.chose_cam)
        self.rtspButton.clicked.connect(self.chose_rtsp)

        self.yolo_comboBox.currentTextChanged.connect(self.change_model_yolo)
        self.reid_comboBox.currentTextChanged.connect(self.change_model_reid)
        self.confSpinBox_yolo.valueChanged.connect(lambda x: self.change_val(x, 'confSpinBox_yolo'))
        self.confSlider_yolo.valueChanged.connect(lambda x: self.change_val(x, 'confSlider_yolo'))
        self.confSpinBox_reid.valueChanged.connect(lambda x: self.change_val(x, 'confSpinBox_reid'))
        self.confSlider_reid.valueChanged.connect(lambda x: self.change_val(x, 'confSlider_reid'))
      
        self.saveCheckBox.clicked.connect(self.is_save)
        self.autoDetectCheckBox.clicked.connect(self.is_auto_detect)

        self.runButton.clicked.connect(self.run_or_continue)
        self.stopButton.clicked.connect(self.stop)

    def load_setting(self):
        self.yolo_comboBox.clear()
        self.reid_comboBox.clear()
        self.yolo_list = os.listdir('./weight/v8')
        self.reid_list = os.listdir("./weight/strongreid")
        self.pt_list = [file for file in self.yolo_list if file.endswith('.pt')]
        self.pth_list = [file for file in self.reid_list if file.endswith('.pth')]
        self.pt_list.sort(key = lambda x: os.path.getsize('./weight/v8/'+x))
        self.pth_list.sort(key = lambda x: os.path.getsize('./weight/strongreid/'+x))
        self.yolo_comboBox.addItems(self.pt_list)
        self.reid_comboBox.addItems(self.pth_list)
        
        self.qtimer_search = QTimer(self)
        self.qtimer_search.timeout.connect(lambda: self.search_pt())
        self.qtimer_search.start(2000)

        self.det_thread = DetThread()
        self.det_thread.weights_yolo = "./weight/v8/%s" % self.yolo_comboBox.currentText()
        self.det_thread.weights_reid = "./weight/strongreid/%s" % self.reid_comboBox.currentText()
        self.det_thread.source = '0'
        self.det_thread.target = './runs/target_person/csl.png'

        self.det_thread.percent_length = self.progressBar.maximum()
        self.det_thread.send_raw.connect(lambda x: self.show_image(x, self.raw_video))
        self.det_thread.send_img.connect(lambda x: self.show_image(x, self.out_video))
        self.det_thread.send_statistic_img.connect(lambda x: self.show_result_img(x,self.result_label))
        self.det_thread.send_statistic_text.connect(lambda x: self.show_result_text(x,self.resutl_label2))

        self.det_thread.send_msg.connect(lambda x: self.show_msg(x))
        self.det_thread.send_percent.connect(lambda x: self.progressBar.setValue(x))
        self.det_thread.send_fps.connect(lambda x: self.fps_label.setText(x))
        
        config_file = 'config/setting.json'
        if not os.path.exists(config_file):
            yolo_conf = 0.25
            reid_conf = 0.45
            savecheck = 0
            autodetect = 0
            new_config = {
                        "yolo_conf": 0.25,
                        "reid_conf": 0.45,
                        "savecheck": 0,
                        "autodetect": 0 }
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(new_json)
        else:
            config = json.load(open(config_file, 'r', encoding='utf-8'))
            if len(config) != 4:
                yolo_conf = 0.55
                reid_conf = 0.60
                savecheck = 0
                autodetect = 0
            else:
                yolo_conf = config['yolo_conf']
                reid_conf = config['reid_conf']
                savecheck = config['savecheck']
                autodetect = config['autodetect']
        self.confSpinBox_yolo.setValue(yolo_conf)
        self.confSpinBox_reid.setValue(reid_conf)
        self.saveCheckBox.setCheckState(savecheck)
        self.autoDetectCheckBox.setCheckState(autodetect)
        self.is_save()
        self.is_auto_detect()

    def search_pt(self):
        pt_list = os.listdir('./weight/v8')
        pth_list = os.listdir("./weight/strongreid")
        pt_list = [file for file in pt_list if file.endswith('.pt')]
        pth_list = [file for file in pth_list if file.endswith('.pth')]
        pt_list.sort(key = lambda x: os.path.getsize('./weight/v8/'+x))
        pth_list.sort(key = lambda x: os.path.getsize('./weight/strongreid/'+x))
        if pt_list != self.pt_list or pth_list != self.pth_list:
            self.pt_list = pt_list
            self.pth_list = pth_list
            self.yolo_comboBox.clear()
            self.reid_comboBox.clear()
            self.yolo_comboBox.addItems(self.pt_list)
            self.reid_comboBox.addItems(self.pth_list)

    def open_target(self):
        config_file = 'config/target.json'
        config = json.load(open(config_file, 'r', encoding='utf-8'))
        open_fold = config['open_fold']
        if not os.path.exists(open_fold):
            open_fold = os.getcwd()
        target_path, _ = QFileDialog.getOpenFileName(self, 'load target_image', open_fold, "*.jpg *.png")

        if target_path:
            self.det_thread.target = target_path
            self.statistic_msg('Loaded target_image: {}'.format(os.path.basename(target_path)))
            self.target_label.setPixmap(QPixmap(target_path).scaled(80, 100))

            config['open_fold'] = os.path.dirname(target_path)
            config_json = json.dumps(config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(config_json)
            self.stop()

    def chose_file(self):
        config_file = 'config/file.json'
        config = json.load(open(config_file, 'r', encoding='utf-8'))
        open_fold = config['open_fold']
        if not os.path.exists(open_fold):
            open_fold = os.getcwd()
        name, _ = QFileDialog.getOpenFileName(self, 'Video/image', open_fold, "Pic File(*.mp4 *.mkv *.avi *.flv *.jpg *.png)")
        if name:
            self.det_thread.source = name
            self.statistic_msg('Loaded file: {}'.format(os.path.basename(name)))
            config['open_fold'] = os.path.dirname(name)
            config_json = json.dumps(config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(config_json)
            self.stop()

    def chose_cam(self):
        try:
            self.stop()
            MessageBox(title='Tips', text='Loading camera', time=2000, auto=True).exec_()
            _, cams = Camera().get_cam_num()
            
            popMenu = QMenu()
            popMenu.setFixedWidth(self.cameraButton.width())
            popMenu.setStyleSheet('''
                            QMenu {
                            font-size: 16px;
                            font-family: "Microsoft YaHei UI";
                            font-weight: light;
                            color:white;
                            padding-left: 5px;
                            padding-right: 5px;
                            padding-top: 4px;
                            padding-bottom: 4px;
                            border-style: solid;
                            border-width: 0px;
                            border-color: rgba(255, 255, 255, 255);
                            border-radius: 3px;
                            background-color: rgba(200, 200, 200,50);}
                            ''')
            
            for cam in cams:
                exec("action_%s = QAction('%s')" % (cam, cam))
                exec("popMenu.addAction(action_%s)" % cam)
            
            x = self.groupBox_5.mapToGlobal(self.cameraButton.pos()).x()
            y = self.groupBox_5.mapToGlobal(self.cameraButton.pos()).y()
            y = y + self.cameraButton.frameGeometry().height()
            pos = QPoint(x, y)
            
            action = popMenu.exec_(pos)
            
            if action:
                self.det_thread.source = action.text()
                self.statistic_msg('Loading camera：{}'.format(action.text()))
        
        except Exception as e:
            self.statistic_msg('%s' % e)

    def chose_rtsp(self):
        self.rtsp_window = Window()
        config_file = 'config/ip.json'
        if not os.path.exists(config_file):
            ip = "rtsp://admin:admin888@192.168.1.67:555"
            new_config = {"ip": ip}
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(new_json)
        else:
            config = json.load(open(config_file, 'r', encoding='utf-8'))
            ip = config['ip']
        self.rtsp_window.rtspEdit.setText(ip)
        self.rtsp_window.show()
        self.rtsp_window.rtspButton.clicked.connect(lambda: self.load_rtsp(self.rtsp_window.rtspEdit.text()))

    def load_rtsp(self, ip):
        try:
            self.stop()
            MessageBox(title='Tips', text='Loading rtsp stream', time=1000, auto=True).exec_()
            self.det_thread.source = ip
            self.statistic_msg('Loading rtsp：{}'.format(ip))
            self.rtsp_window.close()
        except Exception as e:
            self.statistic_msg('%s' % e)

    def run_or_continue(self):
        self.det_thread.jump_out = False
        if self.runButton.isChecked():
            self.saveCheckBox.setEnabled(False)
            self.yolo_comboBox.setEnabled(False)
            self.reid_comboBox.setEnabled(False)
            self.targetbox.setEnabled(False)
            self.progressBar.setEnabled(False)
        
            self.det_thread.is_continue = True 
            if not self.det_thread.isRunning():
                self.det_thread.start()
            source = os.path.basename(self.det_thread.source)
            source = 'camera' if source.isnumeric() else source
            self.statistic_msg('Detecting >> yolo_model：{} >> reid_model：{} >> target：{} >> file：{}'.
                               format(os.path.basename(self.det_thread.weights_yolo),
                                      os.path.basename(self.det_thread.weights_reid),
                                      os.path.basename(self.det_thread.target),
                                      source))
        else:
            self.det_thread.is_continue = False
            self.progressBar.setEnabled(True)
            self.statistic_msg('Pause')

    def stop(self):
        self.det_thread.jump_out = True
        self.saveCheckBox.setEnabled(True)
        self.yolo_comboBox.setEnabled(True)
        self.reid_comboBox.setEnabled(True)
        self.targetbox.setEnabled(True)
        self.progressBar.setEnabled(True)

    def is_save(self):
        if self.saveCheckBox.isChecked():
            self.det_thread.save_fold = './result'
        else:
            self.det_thread.save_fold = None

    def is_auto_detect(self):
        if self.autoDetectCheckBox.isChecked():
            self.confSpinBox_yolo.setEnabled(False)
            self.confSlider_yolo.setEnabled(False)
            self.confSpinBox_reid.setEnabled(False)
            self.confSlider_reid.setEnabled(False)
            self.confSpinBox_yolo.setValue(0.2)
            self.confSpinBox_reid.setValue(0.0)
            self.det_thread.auto_detect = True
            self.statistic_label.setStyleSheet("color: green; font-weight: bold;")
            self.statistic_msg('自动检测已启用')
            self.autoDetectCheckBox.setText("自动检测模式（已启用）")
        else:
            self.confSpinBox_yolo.setEnabled(True)
            self.confSlider_yolo.setEnabled(True)
            self.confSpinBox_reid.setEnabled(True)
            self.confSlider_reid.setEnabled(True)
            self.det_thread.auto_detect = False
            self.statistic_label.setStyleSheet("color: black; font-weight: normal;")
            self.statistic_msg('自动检测已禁用')
            self.autoDetectCheckBox.setText("自动检测模式（已禁用）")

    def mousePressEvent(self, event):
        self.m_Position = event.pos()
        if event.button() == Qt.LeftButton:
            if 0 < self.m_Position.x() < self.groupBox.pos().x() + self.groupBox.width() and \
                    0 < self.m_Position.y() < self.groupBox.pos().y() + self.groupBox.height():
                self.m_flag = True

    def mouseMoveEvent(self, QMouseEvent):
        if Qt.LeftButton and self.m_flag:
            self.move(QMouseEvent.globalPos() - self.m_Position)

    def mouseReleaseEvent(self, QMouseEvent):
        self.m_flag = False
    
    def max_or_restore(self):
        if self.maxButton.isChecked():
            self.showMaximized()
        else:
            self.showNormal()
    
    def closeEvent(self, event):
        self.det_thread.jump_out = True
        config_file = 'config/setting.json'
        config = dict()
        config['yolo_conf'] = self.confSpinBox_reid.value()
        config['reid_conf'] = self.confSpinBox_yolo.value()
        config['savecheck'] = self.saveCheckBox.checkState()
        config['autodetect'] = self.autoDetectCheckBox.checkState()
        config_json = json.dumps(config, ensure_ascii=False, indent=2)
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(config_json)
        MessageBox(title='提示', text='正在关闭程序...', time=2000, auto=True).exec_()
        sys.exit(0)

    def change_val(self, x, flag):
        if flag == 'confSpinBox_yolo':
            self.confSlider_yolo.setValue(int(x*100))
        elif flag == 'confSlider_yolo':
            self.confSpinBox_yolo.setValue(x/100)
            self.det_thread.conf_thres_yolo = x/100
        elif flag == 'confSpinBox_reid':
            self.confSlider_reid.setValue(int(x*100))
        elif flag == 'confSlider_reid':
            self.confSpinBox_reid.setValue(x/100)
            self.det_thread.conf_thres_reid = x/100
 
        else:
            pass

    def change_model_yolo(self, x):
        self.model_type= self.yolo_comboBox.currentText()
        self.det_thread.weights_yolo = "./weight/v8/%s" % self.model_type
        self.statistic_msg('Change yolo model to %s' % x)

    def change_model_reid(self, x):
        self.model_type= self.reid_comboBox.currentText()
        self.det_thread.weights_reid = "./weight/strongreid/%s" % self.model_type
        self.statistic_msg('Change reid model to %s' % x)

    @staticmethod
    def show_image(img_src, Qlabel):
        try:
            ih, iw, _ = img_src.shape
            w = Qlabel.geometry().width()
            h = Qlabel.geometry().height()
            if iw/w > ih/h:
                scal = w / iw
                nw = w
                nh = int(scal * ih)
                img_src_ = cv2.resize(img_src, (nw, nh))
            else:
                scal = h / ih
                nw = int(scal * iw)
                nh = h
                img_src_ = cv2.resize(img_src, (nw, nh))

            frame = cv2.cvtColor(img_src_, cv2.COLOR_BGR2RGB)
            img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[2] * frame.shape[1],
                         QImage.Format_RGB888)
            Qlabel.setPixmap(QPixmap.fromImage(img))
        except Exception as e:
            print(repr(e))
    
    @staticmethod
    def show_result_img(img_src, Qlabel):
        try:
            ih, iw, _ = img_src.shape
            w = Qlabel.geometry().width()
            h = Qlabel.geometry().height()
            if iw/w > ih/h:
                scal = w / iw
                nw = w
                nh = int(scal * ih)
                img_src_ = cv2.resize(img_src, (nw, nh))
            else:
                scal = h / ih
                nw = int(scal * iw)
                nh = h
                img_src_ = cv2.resize(img_src, (nw, nh))

            frame = cv2.cvtColor(img_src_, cv2.COLOR_BGR2RGB)
            img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[2] * frame.shape[1],
                         QImage.Format_RGB888)
            Qlabel.setPixmap(QPixmap.fromImage(img))
        except Exception as e:
            print(repr(e))

    @staticmethod
    def show_result_text(msg, Qlabel):
        Qlabel.setText(f'target: {msg} frame')
  
    def statistic_msg(self, msg):
        self.statistic_label.setText(msg)
    
    def show_msg(self, msg):
        self.runButton.setChecked(Qt.Unchecked)
        self.statistic_msg(msg)
        if msg == "Finished":
            self.saveCheckBox.setEnabled(True) 