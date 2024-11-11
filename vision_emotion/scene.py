import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import cv2


# 假设你已经有计算面部特征的函数
def calculate_face_ratio(image):
    # 示例：假设返回的比值为随机数
    eyebrow_eye_ratio = np.random.uniform(1.5, 2.5)  # 随机模拟眉眼比
    mouth_eye_ratio = np.random.uniform(0.5, 1.5)  # 随机模拟嘴眼比
    return eyebrow_eye_ratio, mouth_eye_ratio


class EmotionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Emotion Detection")
        self.root.geometry("800x600")  # 窗口大小

        # 载入背景图片
        self.background_image = Image.open("123.jpg")  # 替换为你的背景图片路径

        # 调整背景图大小，使其填充整个窗口
        self.bg_image = self.background_image.resize((800, 600))  # 确保背景图填充窗口
        self.bg_image = ImageTk.PhotoImage(self.bg_image)

        # 创建一个Canvas用来放置背景图
        self.canvas = tk.Canvas(self.root, width=800, height=600)  # 窗口尺寸调整为800x600
        self.canvas.pack(fill="both", expand=True)

        # 在Canvas上放置背景图
        self.canvas.create_image(0, 0, anchor="nw", image=self.bg_image)

        # 右侧的UI区域
        self.right_frame = tk.Frame(self.root, width=250, bg="lightblue", bd=5, relief="solid", padx=10, pady=10)
        self.right_frame.place(x=600, y=0, relheight=1)  # 锚点设定使其固定在右侧

        # 给右侧框架添加圆角效果（可以通过添加Canvas绘制圆角来实现）

        # 右侧显示面部特征信息
        self.eyebrow_eye_label = tk.Label(self.right_frame, text="Eyebrow-Eye Ratio: ", font=("Arial", 14), fg="black",
                                          bg="lightblue")
        self.eyebrow_eye_label.pack(pady=10)

        self.mouth_eye_label = tk.Label(self.right_frame, text="Mouth-Eye Ratio: ", font=("Arial", 14), fg="black",
                                        bg="lightblue")
        self.mouth_eye_label.pack(pady=10)

        # 显示面部特征的标签
        self.eyebrow_eye_value = tk.Label(self.right_frame, text="Calculating...", font=("Arial", 12), fg="blue",
                                          bg="lightblue")
        self.eyebrow_eye_value.pack(pady=5)

        self.mouth_eye_value = tk.Label(self.right_frame, text="Calculating...", font=("Arial", 12), fg="blue",
                                        bg="lightblue")
        self.mouth_eye_value.pack(pady=5)

        # 模拟处理图片的按钮
        self.process_button = tk.Button(self.right_frame, text="Process Image", font=("Arial", 14),
                                        command=self.update_ratios, bg="lightgreen", fg="black", relief="raised")
        self.process_button.pack(pady=20)

        # 设置摄像头捕获
        self.video_source = 0  # 默认摄像头
        self.cap = cv2.VideoCapture(self.video_source)
        self.update_camera()

    def update_ratios(self):
        # 假设在此获取到实际图片和计算面部特征的代码
        # 这里使用随机数代替实际的面部比率计算
        eyebrow_eye_ratio, mouth_eye_ratio = calculate_face_ratio(None)  # 图片传入实际代码

        # 更新UI标签
        self.eyebrow_eye_value.config(text=f"{eyebrow_eye_ratio:.2f}")
        self.mouth_eye_value.config(text=f"{mouth_eye_ratio:.2f}")

    def update_camera(self):
        # 捕获摄像头的帧
        ret, frame = self.cap.read()
        if ret:
            # 转换颜色从BGR到RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 转换为PIL图像
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))

            # 在Canvas上显示图像
            self.canvas.create_image(400, 300, image=self.photo)

        # 继续更新摄像头画面
        self.root.after(10, self.update_camera)

    def on_closing(self):
        # 关闭时释放摄像头
        self.cap.release()
        self.root.quit()


if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionApp(root)

    # 捕捉窗口关闭事件，确保摄像头在关闭时释放
    root.protocol("WM_DELETE_WINDOW", app.on_closing)

    root.mainloop()
