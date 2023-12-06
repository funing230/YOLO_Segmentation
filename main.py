from ultralytics import YOLO
import torch
from PIL import Image
# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def train():
    # Load a model
    model = YOLO('yolov8n-seg.pt').to(device)  # load a pretrained model (recommended for training)

    # Train the model
    model.train(data='coco8-seg.yaml', epochs=500, imgsz=640)

    results = model("C:/Users/ssel512/PycharmProjects/datasets/coco8-seg/images/val/000000000061.jpg")

    print("----------------------------------------------------------------------")
    # # 处理结果列表
    for result in results:
        boxes = result.boxes  # 边界框输出的 Boxes 对象
        print("Prediction boxes:", boxes)
        masks = result.masks  # 分割掩码输出的 Masks 对象
        print("Prediction masks:", masks)
        keypoints = result.keypoints  # 姿态输出的 Keypoints 对象
        print("Prediction keypoints:", keypoints)
        probs = result.probs  # 分类输出的 Probs 对象
        print("Prediction probs:", probs)
    print("Prediction results:", results)
    print("----------------------------------------------------------------------")
    # 展示结果
    for r in results:
        im_array = r.plot()  # 绘制包含预测结果的BGR numpy数组
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL图像
        im.show()  # 显示图像
        im.save('results_segmentation.jpg')  # 保存图像
if __name__ == '__main__':
    train()
