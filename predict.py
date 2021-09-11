import numpy as np 
import matplotlib.pyplot as plt 
import torch, torchvision 
from torchvision import transforms 
from PIL import Image 
import pandas as pd 
import cv2, time, json, os, pickle, uuid 
from typing import List, Dict, Any, Tuple 
from glob import glob 
import argparse

resize = 224 
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
trans = transforms.Compose([
                            transforms.Resize((resize, resize)),
                            transforms.ToTensor(), 
                            transforms.Normalize(mean, std)
])

label = ['__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

def load_model():
    model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
    model.eval()
    return model 

def detect(input: torch.Tensor, net: object) -> np.ndarray:
    with torch.no_grad():
        out = net(input) # (3, 224, 224) -> (21, 224, 224)
        out = out["out"][0].argmax(0).cpu().detach().numpy() # (w, h) dtype int 
    return out 


def main(img1: str, img2: str, display_label: List[str]):
    # 画像変換
    main_img = Image.open(img1)
    main_img_ = np.array(main_img)
    h, w, _ = main_img_.shape # メイン画像から高さ幅を保持する
    back_ground = Image.open(img2)
    back_ground = back_ground.resize((w, h))
    # セグメンテーション推論
    main_tensor = trans(main_img).unsqueeze(0)
    net = load_model()
    output = detect(main_tensor, net)
    # 指定されたラベルのみ抽出する
    labels = []
    for l in display_label:
        idx = label.index(l)
        labels.append(idx)
    # 指定するラベルの分だけマスクの作成
    # セグメントされた分だけ０・１に変換する
    mask = np.zeros((224, 224))
    for l in labels:
        mask += (output == l).astype(float)
    mask = cv2.resize(mask, (w, h))
    mask = cv2.cvtColor(np.uint8(mask), cv2.COLOR_GRAY2RGB) # (w, h, c)
    # 合成画像の作成
    # マスクは0か1で構成されているので積をとることで抜き出す
    composite = np.uint8((mask*main_img_)+(back_ground*(1.0-mask)))
    plt.imshow(composite)
    plt.xticks([])
    plt.yticks([])

    id = uuid.uuid4()
    os.makedirs("result/", exist_ok=True)
    plt.savefig(f"./result/img{str(id)[:2]}.png")
    print(f"result/img{str(id)[:2]}に画像を保存しました。")

# parser 
parser = argparse.ArgumentParser(description='メインの画像と背景画像を挿入してください。')
parser.add_argument('--main', help='main image file path', type=str, default="./img/sample/1.jpg")
parser.add_argument('--background', help='background image file path', type=str, default="./img/sample/2.jpg")
parser.add_argument('--label', help='segmentation labels name ', type=list, default=["person", "cow", "bird"])
args = parser.parse_args()
main_ = str(args.main)
background = str(args.background)
label_list = args.label
main(main_, background, label_list)