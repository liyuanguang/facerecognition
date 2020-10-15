import cv2


def CatchVideo(window_name, camera_idx):
    cv2.namedWindow(window_name)

    # 视频来源，可以选择摄像头或者视频
    # 参数是0，表示打开笔记本的内置摄像头，参数是视频文件路径则打开视频，如vc = cv2.VideoCapture("../testi.mp4")
    cap = cv2.VideoCapture(camera_idx)

    # 使用人脸识别分类器（这里填你自己的OpenCV级联分类器地址）
    classfier = cv2.CascadeClassifier("venv/Lib/site-packages/cv2/data/haarcascade_frontalface_alt2.xml")

    # 识别出人脸后要画的边框的颜色，RGB格式
    color = (0, 255, 0)

    num = 0
    while cap.isOpened():
        ok, frame = cap.read()  # 读取一帧数据
        if not ok:
            break

            # 将当前帧转换成灰度图像
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 人脸检测，1.2和3分别为图片缩放比例和需要检测的有效点数，32*32为最小检测的图像像素
        faceRects = classfier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
        if len(faceRects) > 0:  # 大于0则检测到人脸
            for faceRect in faceRects:  # 框出每一张人脸
                x, y, w, h = faceRect

                # 将当前帧保存为图片
                img_path = 'image/train/lygs'
                img_name = '%s/%d.jpg' % (img_path, num)
                image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
                cv2.imwrite(img_name, image)

                # 画出矩形框
                cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 2)

                # 显示当前捕捉到了多少人脸图片了，这样站在那里被拍摄时心里有个数，不用两眼一抹黑傻等着
                num += 1
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, 'num:%d' % (num), (x + 30, y + 30), font, 1, (255, 0, 255), 4)

                # 显示图像
        cv2.imshow(window_name, frame)

        # cv2.waitKey()等待键盘输入
        # 参数是1，表示延时1ms切换到下一帧图像
        # 参数为0，如cv2.waitKey(0),只显示当前帧图像，相当于视频暂停
        # 参数过大如cv2.waitKey(1000)，会因为延时过久而感觉到卡顿
        # c得到的是键盘输入的ASCII码，esc键对应的ASCII码是27，即当按esc键是if条件句成立
        c = cv2.waitKey(10)
        if c & 0xFF == ord('q'):  # 按q退出
            break

            # 释放摄像头并销毁所有窗口
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    CatchVideo("Camera", 0)
