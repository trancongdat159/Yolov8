import sys
import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
from tkinter import Tk, Label, Button, Text, filedialog, messagebox, Frame
from PIL import Image, ImageTk
import math

# Thêm đường dẫn đến thư mục chứa thư viện Sort
sys.path.append(r"C:\Users\ledin\sort_module")

# Import lớp Sort sau khi thêm đường dẫn
from sort import Sort

# Danh sách các lớp
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

# Hàm xử lý video
def process_video(file_path, mask_path, video_label, result_text):
    cap = cv2.VideoCapture(file_path)
    model = YOLO('yolov8l.pt')
    tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

    # Đọc mask
    mask = cv2.imread(mask_path, 0)
    if mask is None:
        messagebox.showerror("Lỗi", "Không thể đọc mask. Đảm bảo đường dẫn đúng.")
        return

    vehicle_count = {
        "car": 0,
        "motorbike": 0,
        "bicycle": 0,
        "truck": 0
    }
    counted_ids = {key: set() for key in vehicle_count.keys()}
    object_colors = {}

    limits = [0, 500, 1200, 500]  # Đường giới hạn đầu tiên
    limits_2 = [limits[0], limits[1] + 50, limits[2], limits[3] + 50]  # Đường giới hạn thứ hai

    def update_frame():
        success, img = cap.read()
        if not success:
            cap.release()
            return

        # Thay đổi kích thước của mask cho phù hợp với img và chuyển sang 3 kênh
        mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))
        mask_resized = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)

        # Áp dụng mask
        imgRegion = cv2.bitwise_and(img, mask_resized)

        results = list(model(imgRegion, stream=True))
        detections = np.empty((0, 5))

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                currentClass = classNames[cls]

                if currentClass in vehicle_count.keys() and conf > 0.3:
                    detections = np.vstack((detections, [x1, y1, x2, y2, conf]))

                    # Vẽ khung bao quanh đối tượng
                    cvzone.cornerRect(img, (x1, y1, x2 - x1, y2 - y1), l=9, rt=2, colorR=(255, 0, 255))
                    cv2.putText(img, f'{currentClass}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        resultsTracker = tracker.update(detections)

        for result in resultsTracker:
            x1, y1, x2, y2, id = result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cx, cy = x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2

            # Vẽ chấm trung tâm
            if id not in object_colors:
                object_colors[id] = (255, 0, 255)  # Màu tím cho chấm trung tâm ban đầu

            cv2.circle(img, (cx, cy), 5, object_colors[id], cv2.FILLED)

            # Kiểm tra và đếm xe
            for box in results[0].boxes:
                bx1, by1, bx2, by2 = box.xyxy[0]
                if (abs(bx1 - x1) < 10 and abs(by1 - y1) < 10):
                    currentClass = classNames[int(box.cls[0])]
                    break

            if currentClass in vehicle_count.keys():
                if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
                    if id not in counted_ids[currentClass]:
                        counted_ids[currentClass].add(id)
                        vehicle_count[currentClass] += 1
                        object_colors[id] = (0, 0, 255)  # Chuyển sang màu đỏ khi được đếm
                elif limits_2[0] < cx < limits_2[2] and limits_2[1] - 15 < cy < limits_2[1] + 15:
                    if id not in counted_ids[currentClass]:
                        counted_ids[currentClass].add(id)
                        vehicle_count[currentClass] += 1
                        object_colors[id] = (0, 0, 255)  # Chuyển sang màu đỏ khi được đếm

        # Vẽ đường kẻ vạch
        cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)
        cv2.line(img, (limits_2[0], limits_2[1]), (limits_2[2], limits_2[3]), (0, 255, 255), 5)

        # Cập nhật kết quả
        result_text.delete("1.0", "end")
        for vehicle_type, count in vehicle_count.items():
            result_text.insert("end", f"{vehicle_type.capitalize()}: {count}\n")

        # Hiển thị video
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(img_pil)
        video_label.imgtk = img_tk
        video_label.configure(image=img_tk, width=img_tk.width(), height=img_tk.height())

        # Lặp lại
        video_label.after(10, update_frame)

    update_frame()

# Hàm giao diện chính
def start_interface():
    root = Tk()
    root.title("Nhận diện phương tiện giao thông")
    root.geometry("1200x700")

    Label(root, text="Đường dẫn video:").grid(row=0, column=0, sticky="e")
    video_path_entry = Text(root, width=50, height=1)
    video_path_entry.grid(row=0, column=1, padx=10)

    Label(root, text="Đường dẫn mask:").grid(row=1, column=0, sticky="e")
    mask_path_entry = Text(root, width=50, height=1)
    mask_path_entry.grid(row=1, column=1, padx=10)

    def browse_video():
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi")])
        video_path_entry.delete("1.0", "end")
        video_path_entry.insert("1.0", file_path)

    def browse_mask():
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg")])
        mask_path_entry.delete("1.0", "end")
        mask_path_entry.insert("1.0", file_path)

    Button(root, text="Chọn video", command=browse_video).grid(row=0, column=2)
    Button(root, text="Chọn mask", command=browse_mask).grid(row=1, column=2)

    video_frame = Frame(root, width=640, height=480, bg="black")
    video_frame.grid(row=2, column=0, columnspan=2, pady=20)
    video_frame.grid_propagate(False)
    video_label = Label(video_frame)
    video_label.pack(fill="both", expand=True)

    result_text = Text(root, width=40, height=20)
    result_text.grid(row=2, column=2, padx=10)

    def start_processing():
        video_path = video_path_entry.get("1.0", "end").strip()
        mask_path = mask_path_entry.get("1.0", "end").strip()
        if not video_path or not mask_path:
            messagebox.showwarning("Cảnh báo", "Hãy chọn cả video và mask!")
            return
        process_video(video_path, mask_path, video_label, result_text)

    Button(root, text="Bắt đầu xử lý", command=start_processing).grid(row=3, column=1, pady=20)

    root.mainloop()

# Chạy giao diện
start_interface()