import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import json
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# 训练图片和标签
images = []
model_lables = {}


# 加载路径下的所有图片
def load_images(path):
    for file_or_dir in os.listdir(path):
        child_path = os.path.join(path, file_or_dir)
        abs_path = os.path.abspath(child_path)  # 绝对路径
        if os.path.isdir(abs_path):  # 判断是否为目录
            load_images(abs_path)
        else:  # file
            if file_or_dir.endswith('.jpg'):
                image = mpimg.imread(abs_path)
                images.append(image)

    return images


# 加载数据集和标签
# 以上级目录为标签
def load_dataset_lable(path):
    images = []
    labels = []
    model_lables.clear()

    if os.path.isdir(path):
        num = 0
        listdir = os.listdir(train_dir)
        for dir in listdir:
            train_type_dir = os.path.join(train_dir, dir)
            train_type_list = os.listdir(train_type_dir)
            images.append(train_type_list)
            labels.append(num)
            model_lables[dir] = num
            num += 1
    return images, labels


# 加载模型标签
def load_model_lable(fiel_path):
    if not fiel_path or not os.path.isfile(fiel_path):
        return

    with open(fiel_path, 'r', encoding='utf-8') as file:  # 读文件
        content = file.read()
        data = {}
        if len(content) > 0:
            data = json.loads(content)

        file.close()
        return data


# 保存模型标签
def save_model_lable(fiel_path, obj_json):
    if not obj_json or not os.path.isfile(fiel_path):
        return

    with open(fiel_path, "w+") as file:  # 写文件
        json.dump(obj_json, file, indent=4, separators=(",", " : "), ensure_ascii=True)
        file.close()
    print("JSON Mapping for the model lable saved to ", json_path)


# 模型训练
# train_dir 训练数据集路径
# json_path 模型标签json文件路径
def model_train(train_dir, json_path):
    isdir = os.path.isdir(train_dir)
    if not isdir:
        return

    # 训练参数
    batch_size = 128
    epochs = 15
    IMG_HEIGHT = 200
    IMG_WIDTH = 200

    # train_images, train_labels = load_dataset_lable(train_dir)
    # save_model_lable(json_path, model_lables)

    # 训练集
    # 利用实时数据批量生成张量图像数据
    # 对训练图像应用了重新缩放，45度旋转，宽度偏移，高度偏移，水平翻转和缩放增强。
    image_gen_train = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=45,
        width_shift_range=.15,
        height_shift_range=.15,
        horizontal_flip=True,
        zoom_range=0.5
    )

    train_data_gen = image_gen_train.flow_from_directory(batch_size=batch_size,
                                                         directory=train_dir,
                                                         shuffle=True,
                                                         target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                         class_mode='binary')

    save_model_lable(json_path, train_data_gen.class_indices)

    model = Sequential([
        Conv2D(16, 3, padding='same', activation='relu',
               input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        MaxPooling2D(),
        Dropout(0.2),
        Conv2D(32, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(64, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Dropout(0.2),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    # 编译模型
    # 指定损失、指标和优化器
    # optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-3)
    model.compile()

    # 模型总结
    model.summary()

    # 模型训练
    # 使用fit方法来训练网络。
    # 为固定的时间段（数据集上的迭代）训练模型
    # x训练数据集
    # y训练数据集标签
    history = model.fit(
        train_data_gen,
        batch_size=batch_size,
        epochs=epochs
    )

    model.save("model/model_file.h5")

    # acc = history.history['accuracy']
    # val_acc = history.history['val_accuracy']
    #
    # loss = history.history['loss']
    # val_loss = history.history['val_loss']
    #
    # epochs_range = range(epochs)
    #
    # plt.figure(figsize=(15, 5))
    # plt.subplot(1, 2, 1)
    # plt.plot(epochs_range, acc, label='Training Accuracy')
    # plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    # plt.legend(loc='lower right')
    # plt.title('Training and Validation Accuracy')
    #
    # plt.subplot(1, 2, 2)
    # plt.plot(epochs_range, loss, label='Training Loss')
    # plt.plot(epochs_range, val_loss, label='Validation Loss')
    # plt.legend(loc='upper right')
    # plt.title('Training and Validation Loss')
    # plt.show()

    print("模型训练结束")


if __name__ == '__main__':
    json_path = "model/model_lable.json"
    # data = load_model_lable(json_path)

    train_dir = 'image/train'
    model_train(train_dir, json_path)
