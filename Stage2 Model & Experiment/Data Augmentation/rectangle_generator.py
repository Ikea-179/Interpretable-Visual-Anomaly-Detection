import cv2
import numpy as np
img = np.ones((4,4),dtype=np.uint8)#random.random()方法后面不能加数据类型
#img = np.random.random((3,3)) #生成随机数都是小数无法转化颜色,无法调用cv2.cvtColor函数
img[0,0]=100
img[0,1]=150
img[0,2]=255
bgr_img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
bgr_img[:,:,0] = 0
bgr_img[:,:,1] = 255
bgr_img[:,:,2] = 255
cv2.imwrite("./fminist/"+"rectangle.jpg",bgr_img)