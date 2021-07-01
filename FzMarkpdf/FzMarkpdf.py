import cv2
import numpy as np
import fitz
import os
import pdfplumber
import argparse

'''
FzMarkpdf:
1. 环境依赖
pip install opencv-python
pip install numpy
pip install traits==6.1.1
pip install PyMuPDF
pip install fitz
pip install pdfplumber
2. 使用手册
传入需要进行绘制的pdf文件路径+名称，传入对应拍摄的照片命名规范 pagenum.jpg (1.jpg)
可参照examples文件中的例子
'''

DEBUG_MODE = True

parser = argparse.ArgumentParser(description="FzMarkpdf")
parser.add_argument('--pdfname', '-n', help='pdf文件名称 || The name of pdf.', type=str, required=True)
parser.add_argument('--pdfpath', '-pp', help='pdf文件路径 || The path of pdf.', type=str, default="./")
parser.add_argument('--imagespath', '-p', help='笔迹拍摄图片路径 || The path of images that are marked.', type=str, default="./images/")
parser.add_argument('--scale', '-s', help='打印页面的缩放比例(100缩放 等于 1.0) || The scale of the printed page(100zoom equals 1.0).', type=float, default=1.0)
parser.add_argument('--correction', '-c', help='页面缩放偏移修正([scale_x,y,offset_x,y]) || Page zoom offset corrected([scale_x,y,offset_x,y]).', type=list, default=[1.0, 1.0, 0.0, 0.0])
parser.add_argument('--minbox', '-m', help='将小于此数值的点视为噪点 || Consider points smaller than this value as noise.', type=int, default=600)
args = parser.parse_args()

INPUT_PDF_FILE = args.pdfname  # "MVP Detecting Vulnerabilities using Patch-Enhanced Vulnerability Signatures.pdf"
OUT_PDF_FILE = "Mark_" + INPUT_PDF_FILE
PDF_PATH = args.pdfpath  #"./"
IMAGES_PATH = args.imagespath  # "./images/"
TMP_DIR_NAME = "tmpsavepng"
TMP_PATH = './' + TMP_DIR_NAME + '/'

# 分辨率72像素/英寸时，A4纸尺寸像素596*842
# 分辨率150像素/英寸时，A4纸尺寸像素1240*1754
# 分辨率300像素/英寸时，A4纸尺寸像素2480*3508
# 分辨率改变时，其中的腐蚀与膨胀核也需相应的改变
A4_WIDTH = 596
A4_HEIGHT = 842
PIXEL_VALUE_X = 2480
PIXEL_VALUE_Y = 3508
PIXEL_SCALE = 72  # 72 分辨缩放
PIXEL_SCALE = PIXEL_SCALE / 300  # 72 分辨缩放

# 得到pdf文档的长和宽
with pdfplumber.open(PDF_PATH + INPUT_PDF_FILE) as pdf:
    PDF_WIDTH = float(pdf.pages[0].width)
    PDF_HEIGHT = float(pdf.pages[0].height)

PRINT_SCALE = args.scale  # 打印页面时缩放比例 打印论文阅读时一般会设置缩放比例 将缩放比例填入，可以更精准的将笔迹附加上
PRINT_SCALE = 1.0 / PRINT_SCALE

PAPER_SCALE_X = PDF_WIDTH  # paper与A4纸大小之间存在差异 需要缩放  596*842px 21*29.7cm
PAPER_SCALE_X = A4_WIDTH / PAPER_SCALE_X
PAPER_SCALE_Y = PDF_HEIGHT  # cm
PAPER_SCALE_Y = A4_HEIGHT / PAPER_SCALE_Y

# PAPER_SCALE_X PAPER_SCALE_Y 与 A4纸21*29.7哪个轴差距大修正需要对应加大  扫描时和打印时的白边会导致偏差，进行修正
OFFSET_X = - (A4_WIDTH - PDF_WIDTH)  # 偏移 相当于裁剪掉打印时的白边，使其贴合
OFFSET_Y = - (A4_HEIGHT - PDF_HEIGHT)

# 偏差修正 此处原因尚不知晓
ERROR_CORRECTION_X = 1.08
ERROR_CORRECTION_Y = 1.02
ERROR_OFFSET_X = -24.0
ERROR_OFFSET_Y = -6.0

# 人工修正 缩放 偏移 1.0 1.0 0.0 0.0
SELF_CORRECTION_X = args.correction[0]
SELF_CORRECTION_Y = args.correction[1]
SELF_OFFSET_X = args.correction[2]
SELF_OFFSET_Y = args.correction[3]

if DEBUG_MODE:
    print("PDF_WIDTH", PDF_WIDTH)
    print("PDF_HEIGHT", PDF_HEIGHT)
    print("PRINT_SCALE", PRINT_SCALE)
    print("PAPER_SCALE_X", PAPER_SCALE_X)
    print("PAPER_SCALE_Y", PAPER_SCALE_Y)
    print("OFFSET_X", OFFSET_X)
    print("OFFSET_Y", OFFSET_Y)


PEN_COLOR = {
    'red': [31, 70, 255, 255],
    'blue': [255, 70, 31, 255]}  # Pdf笔迹颜色可定制

MIN_BOX = args.minbox  # 将小于MIN_BOX的图片忽略 去除噪点 600
MAX_BOX = 8699840  # 将大于MIN_BOX的图片忽略 去掉整张纸张的边缘 8699840


def print_logo():
    print("""
 #######################################################
##                                                     ##
##       FFFFFFFFFFF                                   ##
##       FFFFFFFFFF                                    ##
##       FFF                                           ##
##       FFF                                           ##
##       FFFFFFF       zzzzzzzzzz    zzzzzzzzzz        ##
##       FFFFFF              zz            zz          ##
##       FFF               zz            zz            ##
##       FFF             zz            zz              ##
##       FF            zzzzzzzzzz    zzzzzzzzzz        ##
##                                                     ##
##    (^_^)#                                           ##
##      Author:Feng Zhengzhan   Project:FzMarkpdf      ##
##                                                     ##
 #######################################################
    """)
    print(" (^_^)# 欢迎使用烽征战-红色纸质笔迹附加论文PDF项目")
    print(" (^_^)# 项目耗时取决于笔迹数量，给程序点耐心哟 ")


def find_margin(image, pic_path):
    '''
    读取照片中的A4纸张，将其提取出来并缩放到A4纸张大小
    :return:
    '''
    def resizeImg(image, height=900):
        # <class 'numpy.ndarray'> (3648, 2736, 3)
        h, w = image.shape[:2]
        pro = height / h
        size = (int(w * pro), int(height))
        img = cv2.resize(image, size)
        return img

    # 边缘检测
    def getCanny(image):
        # 高斯模糊
        binary = cv2.GaussianBlur(image, (3, 3), 2, 2)
        # 边缘检测
        binary = cv2.Canny(binary, 60, 240, apertureSize=3)
        # 膨胀操作，尽量使边缘闭合
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.dilate(binary, kernel, iterations=1)
        return binary

    def findMaxContour(image):
        # 寻找边缘
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # 计算面积
        max_area = 0.0
        max_contour = []
        for contour in contours:
            currentArea = cv2.contourArea(contour)
            if currentArea > max_area:
                max_area = currentArea
                max_contour = contour
        return max_contour, max_area

    # 多边形拟合凸包的四个顶点
    def getBoxPoint(contour):
        # 多边形拟合凸包
        hull = cv2.convexHull(contour)
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(hull, epsilon, True)
        approx = approx.reshape((len(approx), 2))
        return approx

    # 适配原四边形点集
    def adaPoint(box, pro):
        box_pro = box
        if pro != 1.0:
            box_pro = box / pro
        box_pro = np.trunc(box_pro)
        return box_pro

    # 四边形顶点排序，[top-left, top-right, bottom-right, bottom-left]
    def orderPoints(pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    # 计算长宽
    def pointDistance(a, b):
        return int(np.sqrt(np.sum(np.square(a - b))))

    # 透视变换
    def warpImage(image, box):
        w, h = pointDistance(box[0], box[1]), \
               pointDistance(box[1], box[2])
        dst_rect = np.array([[0, 0],
                             [w - 1, 0],
                             [w - 1, h - 1],
                             [0, h - 1]], dtype='float32')
        M = cv2.getPerspectiveTransform(box, dst_rect)
        warped = cv2.warpPerspective(image, M, (w, h))
        return warped


    ratio = 900 / image.shape[0]
    img = resizeImg(image)
    binary_img = getCanny(img)
    max_contour, max_area = findMaxContour(binary_img)
    boxes = getBoxPoint(max_contour)
    boxes = adaPoint(boxes, ratio)
    boxes = orderPoints(boxes)
    # 透视变化
    warped = warpImage(image, boxes)
    # A4纸张大小
    # warped = cv2.resize(warped, (2480, 3508))
    warped = cv2.resize(warped, (PIXEL_VALUE_X, PIXEL_VALUE_Y))
    if DEBUG_MODE:
        print("[2] 识别裁剪:", TMP_PATH + "find_margin" + pic_path)
        cv2.imwrite(TMP_PATH + "find_margin" + pic_path, warped)

    return warped


def find_draw(image, pic_path):
    '''
    读取拍摄照片并寻找红色笔画轨迹
    :return: 白色蒙版图片
    '''
    '''
    黑 [0, 0, 0] [180, 255, 46]
    灰 [0, 0, 46] [180, 43, 220]
    白 [0, 0, 221] [180, 30, 255]
    红低 [0, 43, 46] [10, 255, 255]
    红高 [156, 43, 46] [180, 255, 255]
    橙 [11, 43, 46] [25, 255, 255]
    黄 [26, 43, 46] [34, 255, 255]
    绿 [35, 43, 46] [77, 255, 255]
    青 [78, 43, 46] [99, 255, 255]
    蓝 [100, 43, 46] [124, 255, 255]
    紫 [125, 43, 46] [155, 255, 255]
    '''
    lower_red = np.array([0, 43, 46])
    upper_red = np.array([10, 255, 255])

    lower_red_high = np.array([130, 43, 46])  # 调整130-156
    upper_red_high = np.array([180, 255, 255])

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    mask_red_high = cv2.inRange(hsv, lower_red_high, upper_red_high)
    mask = cv2.add(mask_red_high, mask_red)

    # colorimg = cv2.imread(r'./colors/white.jpg')
    # colorimg = np.full((3508, 2480, 3), 255, np.uint8)
    colorimg = np.full((PIXEL_VALUE_Y, PIXEL_VALUE_X, 3), 255, np.uint8)
    res = cv2.bitwise_and(colorimg, colorimg, mask=mask)  # 蒙版路径白色

    if DEBUG_MODE:
        print("[3] 红色笔迹蒙版：", TMP_PATH + "find_draw" + pic_path)
        cv2.imwrite(TMP_PATH + "find_draw" + pic_path, res)

    return res


def write_png(file_handle, barcode_file, x, y, w, h, current_page):
    # barcode_file = "2300_2513_160_76.png"
    # define the position (upper-right corner)
    # 页面缩放 * 纸面缩放 * 分辨率 * 修正 + 偏移
    x0 = x * PRINT_SCALE * PIXEL_SCALE * PAPER_SCALE_X * ERROR_CORRECTION_X * SELF_CORRECTION_X + OFFSET_X + ERROR_OFFSET_X + SELF_OFFSET_X
    y0 = y * PRINT_SCALE * PIXEL_SCALE * PAPER_SCALE_Y * ERROR_CORRECTION_Y * SELF_CORRECTION_Y + OFFSET_Y + ERROR_OFFSET_Y + SELF_OFFSET_Y
    x1 = x0 + w * PRINT_SCALE * PIXEL_SCALE * PAPER_SCALE_X * ERROR_CORRECTION_X * SELF_CORRECTION_X
    y1 = y0 + h * PRINT_SCALE * PIXEL_SCALE * PAPER_SCALE_Y * ERROR_CORRECTION_Y * SELF_CORRECTION_Y
    if DEBUG_MODE:
        print(barcode_file, x0, y0, x1, y1, current_page, file_handle)
    image_rectangle = fitz.Rect(x0, y0, x1, y1)  # x0, y0, x1, y1
    # retrieve the first page of the PDF
    pdf_page = file_handle[current_page]
    # add the image
    # 一次性将所有的png图片全部插入，可以附加，注意文件名，不能对同一个文件同时读取和写入
    pdf_page.insertImage(image_rectangle, filename=barcode_file)


def split_whitemask(file_handle, image, colorpen, pic_path_list):

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 300
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))  # 72
    eroded = cv2.erode(image, kernel)

    image = cv2.bitwise_not(eroded)  # 反色

    result = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY)  # 阈值处理

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))  # 300
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))  # 72
    eroded = cv2.erode(thresh, kernel)

    contours, hierarchy = cv2.findContours(eroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    color = (0, 255, 0)

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w*h <= MIN_BOX or w*h >= MAX_BOX:  # 2480 * 3508
            continue
        # print(x, y, w, h)
        if DEBUG_MODE:
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 1)  # 画矩形框
        temp = result[y:(y + h), x:(x + w)]
        filename = str(pic_path_list[0]-1) + '_' + str(x) + '_' + str(y) + '_' + str(w) + '_' + str(h) + '_.png'
        # 背景透明色
        temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGBA)
        sp = temp.shape  # 获取图片维度
        width = sp[0]  # 宽度
        height = sp[1]  # 高度
        for yh in range(height):
            for xw in range(width):
                color_d = temp[xw, yh]  # 遍历图像每一个点，获取到每个点4通道的颜色数据
                if color_d[0] == 255 and color_d[1] == 255 and color_d[2] == 255 and color_d[3] == 255:
                    temp[xw, yh] = [255, 255, 255, 0]
                if color_d[0] == 0 and color_d[1] == 0 and color_d[2] == 0 and color_d[3] == 255:  # 最后一个通道为透明度，如果其值为0，即图像是透明
                    temp[xw, yh] = colorpen  # 则将当前点的颜色设置为白色，且图像设置为不透明

        cv2.imwrite(TMP_PATH + filename, temp)
        write_png(file_handle, TMP_PATH + filename, x, y, w, h, pic_path_list[0])
        if not DEBUG_MODE:
            os.remove(TMP_PATH + filename)

    if DEBUG_MODE:
        print("[4] 识别笔迹：", TMP_PATH + "split_whitemask" + pic_path_list[-1])
        cv2.imwrite(TMP_PATH + "split_whitemask" + pic_path_list[-1], image)


def main():
    print_logo()
    # 建立临时目录
    if not os.path.exists(TMP_DIR_NAME):
        os.mkdir(TMP_DIR_NAME)

    mark_pic = []
    try:
        for file in os.listdir(IMAGES_PATH):
            # 第一个参数为图片页码数，第二个参数为图片名称
            tmp_page = file.split('.')
            tmp_page[0] = int(tmp_page[0])
            tmp_page.append(file)
            mark_pic.append(tmp_page)
        if DEBUG_MODE:
            print("[1] 目录:", mark_pic)
    except Exception as e:
        print("笔迹图片命名错误，参考命名1.jpg 1.png", e)

    file_handle = fitz.open(PDF_PATH + INPUT_PDF_FILE)  # 拿到PDF句柄
    total_num = len(mark_pic)
    current_num = 0
    for each in mark_pic:
        current_num += 1
        print("==> page:{} {}/{}".format(each[0], current_num, total_num))
        image = cv2.imread(str(IMAGES_PATH + each[-1]))
        image = find_margin(image, each[-1])
        image = find_draw(image, each[-1])
        split_whitemask(file_handle, image, PEN_COLOR['red'], each)
    file_handle.save(OUT_PDF_FILE)  # 保存PDF

    if os.path.exists(TMP_DIR_NAME) and not DEBUG_MODE:
        os.rmdir(TMP_DIR_NAME)

    print(" (^_^)# Finish!")

if __name__ == '__main__':
    main()