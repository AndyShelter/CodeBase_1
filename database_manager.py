import pymysql
from datetime import datetime

class DatabaseManager:
    def __init__(self):
        self.connection = None
        try:
            print("正在连接数据库...")
            self.connection = pymysql.connect(
                host='localhost',
                user='86178',
                password='123456',
                database='reid_detection',
                charset='utf8mb4',
                cursorclass=pymysql.cursors.DictCursor
            )
            print("数据库连接成功！")
            self.create_table()
        except pymysql.Error as e:
            print(f"数据库连接错误: {e}")
            print("警告：数据库连接失败，程序将继续运行但不会保存检测结果到数据库")

    def create_table(self):
        if not self.connection:
            return
        try:
            print("正在创建数据表...")
            with self.connection.cursor() as cursor:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS detection_results (
                        id INT AUTO_INCREMENT PRIMARY KEY COMMENT '记录ID',
                        target_id VARCHAR(255) COMMENT '目标行人ID',
                        frame_id VARCHAR(255) COMMENT '视频帧ID',
                        similarity FLOAT COMMENT '相似度分数',
                        detection_time DATETIME COMMENT '检测时间',
                        image_path VARCHAR(255) COMMENT '检测到的行人图片路径',
                        source_type VARCHAR(50) COMMENT '视频源类型(摄像头/视频文件/RTSP)',
                        source_path VARCHAR(255) COMMENT '视频源路径',
                        x INT COMMENT '检测框左上角X坐标',
                        y INT COMMENT '检测框左上角Y坐标',
                        width INT COMMENT '检测框宽度',
                        height INT COMMENT '检测框高度',
                        confidence FLOAT COMMENT '检测置信度',
                        remark TEXT COMMENT '备注信息',
                        INDEX idx_target_id (target_id),
                        INDEX idx_detection_time (detection_time)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='行人重识别检测记录表'
                """)
            self.connection.commit()
            print("数据表创建成功！")
        except pymysql.Error as e:
            print(f"创建表错误: {e}")

    def insert_detection(self, target_id, frame_id, similarity, image_path, detection_box=None, source_type=None, source_path=None):
        if not self.connection:
            print("数据库未连接，无法插入数据")
            return
        
        # 只记录相似度大于0.9的检测结果
        if similarity < 0.9:
            print(f"相似度 {similarity:.2f} 低于阈值0.9，跳过数据库记录")
            return
            
        try:
            print(f"\n正在尝试写入数据库:")
            print(f"- 目标ID: {target_id}")
            print(f"- 帧ID: {frame_id}")
            print(f"- 相似度: {similarity:.2f}")
            print(f"- 检测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"- 视频源类型: {source_type}")
            if detection_box:
                x, y, w, h = detection_box
                print(f"- 位置信息: x={x}, y={y}, 宽={w}, 高={h}")
            
            with self.connection.cursor() as cursor:
                sql = """
                    INSERT INTO detection_results (
                        target_id, frame_id, similarity, detection_time, image_path,
                        source_type, source_path, x, y, width, height, confidence, remark
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                # 解析检测框信息
                x, y, w, h = (0, 0, 0, 0) if detection_box is None else detection_box
                
                # 生成备注信息
                remark = f"检测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, " \
                        f"相似度: {similarity:.2f}, " \
                        f"视频源: {source_type}, " \
                        f"位置: x={x}, y={y}, 宽={w}, 高={h}"
                
                cursor.execute(sql, (
                    target_id,
                    frame_id,
                    similarity,
                    datetime.now(),
                    image_path,
                    source_type,
                    source_path,
                    x,
                    y,
                    w,
                    h,
                    similarity,
                    remark
                ))
            self.connection.commit()
            print("数据库写入成功！")
            print("----------------------------------------")
        except pymysql.Error as e:
            print(f"数据库写入错误: {e}")
            print("----------------------------------------")

    def close(self):
        if self.connection:
            try:
                self.connection.close()
            except:
                pass 