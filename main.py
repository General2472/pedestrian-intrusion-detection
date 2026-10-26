import cv2
import numpy as np
import sys
import csv
import time
from ultralytics import YOLO
from datetime import datetime

#离线写法，只写模型名称如果本地路径检测不到会拉网络下载
model=YOLO(r"D:\embedded-ai\scripts\YOLO_Learning\yolov8n.pt")
cap=cv2.VideoCapture(r'D:\embedded-ai\练手数据集\videos\pedestrians.avi')
log_file = open(r"D:\embedded-ai\练手数据集\logs\events.csv",
                "w", newline="", encoding="utf-8")
writer = csv.writer(log_file)
writer.writerow(["time", "frame", "event", "confidence", "bbox"])
# 注意这个表达 if not (True/False)
if not cap.isOpened():print("打开失败");sys.exit()

#划区域
roi = [(300,150), (500,150), (500,310), (300,310)]  # 示例四个角点

#报警和解除标志
intrusion_frames=0    #连续有人在区域内帧数量
leave_frames=0          #连续无人在区域内帧数量
alarm_state = "NORMAL"  #报警状态

#报警截图编号
alarm_frame_id=0
#切换报警那一刻标志位，用来截图
just_alarm = False
#全局帧索引
frame_index=0
# 循环外累计：
precd_total=0.0
infer_total = 0.0
frame_count = 0
while(True):
    ret,frame=cap.read()
    if not ret:print("读取失败");break #
    #cv2.polylines支持一次画多个多边形，所以第二个参数传numpy数组列表（对比判断函数cv2.pointPolygonTest）
    t1 = time.time()
    cv2.polylines(frame, [np.array(roi, np.int32)], True, (0,0,255), 2)
    t0 = time.time()
    results = model(frame, verbose=False)
    infer_total += time.time() - t0
    frame_count += 1
    res=results[0]
    persons = []
    for box in res.boxes:
        cls = int(box.cls)
        conf = float(box.conf)
        if cls == 0 and conf >= 0.5:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cx=(x1+x2)/2
            cy=(y1+y2)/2
            persons.append((x1, y1, x2, y2, conf,cx,cy))
    #用来储存每帧在危险区域内的人数（box数）
    inside_num=0
    persons_inside=[]
    if persons:
        for x1, y1, x2, y2, conf, cx, cy in persons:
            scan_result = cv2.pointPolygonTest(np.array(roi, np.int32), (cx, cy), False)
            if scan_result > 0:
                persons_inside.append((x1, y1, x2, y2, conf))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
                cv2.putText(frame, f"person {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 1)
                inside_num+=1
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                cv2.putText(frame, f"person {conf:.2f}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 1)

    #报警状态参数累计与清零
    if inside_num: intrusion_frames+=1; leave_frames=0
    else: leave_frames+=1; intrusion_frames=0
    if alarm_state == "NORMAL" and intrusion_frames >= 5:
        alarm_state = "ALARM"
        just_alarm = True
    if alarm_state == "ALARM" and leave_frames >= 10:
        alarm_state = "NORMAL"
        writer.writerow([datetime.now().strftime("%H:%M:%S"), frame_index, "CLEAR", "", ""])

    #显示状态
    if alarm_state == "ALARM":
        color = (0, 0, 255)      # 红
    else:
        color = (0, 255, 0)      # 绿（NORMAL）
    cv2.putText(frame, alarm_state, (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
    #截图切换报警状态帧
    if just_alarm:
        #遍历persons_inside列表的每个元组，比较[4]索引值取最大，返回整个元组
        best_person = max(persons_inside, key=lambda p: p[4])
        best_conf = best_person[4]
        bbox = f"({best_person[0]},{best_person[1]},{best_person[2]},{best_person[3]})"

        cv2.imwrite(rf"D:\embedded-ai\练手数据集\shortcuts\whole\alarm_{alarm_frame_id}.jpg",frame)
        alarm_frame_id+=1
        writer.writerow([
                    datetime.now().strftime("%H:%M:%S"),
                    frame_index,
                    "ALARM",
                    best_conf,          # 这一帧在内人员的置信度（取最高那个）
                    bbox           # 位置，比如 "(x1,y1,x2,y2)"
                ])
        just_alarm = False     # 立刻清掉，下一帧就不再存

    precd_total += time.time()-t1
    #显示画面
    cv2.namedWindow("player", cv2.WINDOW_NORMAL)
    cv2.imshow("player",frame)

    # 1000ms/视频帧率=帧间时间设定，但是实际程序循环跑也需要毫秒时间
    # &0xFF是跨平台写法，只取低八位进行按键检测，去除高八位冗余干扰信息
    if cv2.waitKey(100) & 0xFF == ord('q'):break
    frame_index += 1
avg_ms = infer_total / frame_count * 1000   # 毫秒/帧
pre_avg_ms = precd_total / frame_count * 1000   # 毫秒/帧
print(f"平均推理 {avg_ms:.5f} ms/帧，理论上限 {1000/avg_ms:.3f} FPS")
print(f"单帧总耗时 {pre_avg_ms:.5f} ms/帧，理论上限 {1000/pre_avg_ms:.3f} FPS")
cap.release()
log_file.close()
cv2.destroyAllWindows()
