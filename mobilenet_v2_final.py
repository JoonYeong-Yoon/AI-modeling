# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 14:58:48 2025

@author: human
"""

import numpy as np
import pandas as pd
import glob, os, json, cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tqdm
from sklearn.model_selection import train_test_split

"""
1. 데이터 불러오기 (이미지+JSON 경로 준비)
"""

# 이미지 경로 불러오기 (jpg 확장자)
files = glob.glob("F:/dogDiseaseImage/**/*jpg", recursive=True)

# DataFrame 생성
df = pd.DataFrame({"path": files})

"""
2. 라벨 추출 & 매핑
"""

# 파일명에서 라벨 추출 (예: ..._A1_...jpg → "A1")
df["label"] = [os.path.basename(i).split("_")[-2] for i in files]

# 라벨 매핑 (원하는 라벨 체계에 맞춰 숫자로 변환)
df["label"] = df["label"].map({"A3": 1, "A4": 2, "A6": 3, "A7": 0})
               # A3: 태선화_과다색소 침착, A4: 농포_여드름, A6: 결절_종괴, A7: 무증상 

# 클래스별 데이터 개수 확인
print(df["label"].value_counts())


"""
3. 클래스별 샘플링(데이터 균형 맞춤)  
""" 

export_df = df.groupby("label").apply(lambda x: x).reset_index(drop=True)  # 모든 샘플 사용
# export_df = df.groupby("label").sample(n=2000, random_state=42).reset_index(drop=True)  # 샘플을 2000개로 지정
export_df.label.value_counts()
# 샘플 확인
print(export_df.head())


"""
4. JSON 파싱 → Bounding Box/Polygon 추출
"""

def crop_with_json(path): # 이미지 경로(path)에 대응하는 JSON 파일을 불러와 박스 영역 크롭
    json_file = path.replace("jpg", "json")
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    x, y, w, h = data["labelingInfo"][1]["box"]["location"][0].values()  # Bounding box 좌표 추출
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    img = img[...,::-1]
    #img = cv2.imread(path)[:, :, ::-1]  # BGR → RGB 변환

    export_img = img[y:y+h, x:x+w]      # 박스 영역 크롭
    return img, export_img, (x, y, w, h), data


"""
5. 시각화 (Polygon + Bounding Box 표시)
"""

path, label = export_df.loc[0, :]
img, export_img, (x, y, w, h), data = crop_with_json(path)

# 다각형 좌표 추출
points = np.array(list(data["labelingInfo"][0]["polygon"]["location"][0].values())).reshape(-1, 2)

fig, ax = plt.subplots()
ax.imshow(img)

# 다각형 표시
polygon = patches.Polygon(points, closed=True, fill=False, edgecolor="red", linewidth=2)
ax.add_patch(polygon)

# 박스 표시
rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor="blue", facecolor="none")
ax.add_patch(rect)

plt.title(f"Label: {label}")
plt.show()


"""
6. 크롭 이미지 저장(option #1) : 라벨링 이미지 --> 크롭할 경우

save_dir = "E:/export"
os.makedirs(save_dir, exist_ok=True)
for path, label in tqdm.tqdm(export_df.values, desc="Cropping images"):
    img, export_img, _, _ = crop_with_json(path)
    img.shape
    file_name = os.path.basename(path)
    save_path = os.path.join(save_dir, file_name)
    cv2.imwrite(save_path, export_img[:, :, ::-1])  # RGB → BGR 변환 후 저장
"""
    
    
"""
6. 크롭 이미지 저장(option #2) : 크롭된 이미지 저장 시
"""
save_dir = "E:/export"
os.makedirs(save_dir, exist_ok=True)
import shutil
for path, label in tqdm.tqdm(export_df.values, desc="Cropping images"):
    #saved_path = save_dir+ "/" +"/".join(path.replace("\\","/").split("/")[-2:])
    #saved_dir = os.path.dirname(saved_path)
    #os.makedirs(saved_dir, exist_ok=True)
    #shutil.copy(path,saved_path)
    file_name = os.path.basename(path)
    saved_path = f"{save_dir}/{file_name}"
    if not os.path.exists(saved_path):
        shutil.copy(path,saved_path)

"""
7. 학습용 데이터셋 준비(폴더 구조, 분할 (train/val/test = 8:1:1)
"""

import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. keras 학습용 폴더 구조

base_dir = "E:/keras_dataset"  
os.makedirs(base_dir, exist_ok=True)

files = [f for f in os.listdir(save_dir) if f.endswith(".jpg")]  # 이미지 파일 + 라벨
labels = [f.split("_")[-2] for f in files]
df = pd.DataFrame({"file": files, "label": labels})

df_train, df_temp = train_test_split(df, test_size=0.2,       # Train / Temp(Val+Test) 80:20
                                     stratify=df["label"], random_state=42)
        
df_val, df_test = train_test_split(df_temp, test_size=0.5,    # Temp → Val/Test 50:50 (전체 20%씩)
                                   stratify=df_temp["label"], random_state=42)

splits = {"train": df_train, "validation": df_val, "test": df_test}


for split_name, split_df in splits.items():    # 클래스별 폴더 생성 + 파일 복사
    for label in split_df["label"].unique():
        dir_path = os.path.join(base_dir, split_name, label)
        os.makedirs(dir_path, exist_ok=True)
    for _, row in split_df.iterrows():
        src = os.path.join(save_dir, row["file"])
        dst = os.path.join(base_dir, split_name, row["label"], row["file"])
        shutil.copy(src, dst)

print("Train size:", len(df_train))
df_train["label"].value_counts()
print("Val size:", len(df_val))
df_val["label"].value_counts()
print("Test size:", len(df_test))


"""
8. 모델 학습(PyTorch mobilenet_v2 전이학습)
"""
import tensorflow as tf
import os, json, cv2
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# 1. 모델 구현

inputs = tf.keras.layers.Input((224,224,3))    # 입력층 정의
x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)  # 모델에 맞는 전처리
backbone = tf.keras.applications.MobileNetV2(input_shape = (224,224,3),include_top=False)  # 사전학습 모델 불러오기
x = backbone(x)
x = tf.keras.layers.GlobalAvgPool2D()(x)
outputs = tf.keras.layers.Dense(4, activation='sigmoid')(x)  # 출력층 정의, 활성화함수(sigmoid)
model = tf.keras.Model(inputs, outputs)  # Keras Functional API로 모델 생성
model.summary()


# 2. ImageDataGenerator & 데이터 전처리

IMG_SIZE = 224
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(
    # preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(
    # preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
)

train_generator = train_datagen.flow_from_directory(
    os.path.join(base_dir, "train"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    os.path.join(base_dir, "validation"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# 3. MobileNetV2 모델 정의

inputs = tf.keras.layers.Input((IMG_SIZE, IMG_SIZE, 3))
x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
backbone = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

# Backbone freeze : layer의 weight를 학습하지 않음
for layer in backbone.layers:
    layer.trainable = False
    
x = backbone(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(4, activation='softmax')(x)  # 클래스 4개
model = tf.keras.Model(inputs, outputs)
model.summary()


# 4. 모델 컴파일
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),  # 학습률을 상대적으로 높게 설정
    loss='binary_crossentropy', # binary_crossentropy
    metrics=['accuracy']
)

# 5. 학습(Phase 1) : Dense층만 학습 

EPOCHS_PHASE1 = 10
history1 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS_PHASE1
)


model.save_weights("mymodel.h5")
model.load_weights("mymodel.weights.h5")
model = tf.keras.models.load_model("mymodel.keras")
cm

import numpy as np
from tensorflow.keras.preprocessing import image 
import imagefrom tensorflow.keras.models import load_model

model = load_model("my_model.h5") 