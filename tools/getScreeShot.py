#如何获取亿图的屏幕截图 (不带水印) 很烦 做不到偷图

import time
from PIL import ImageGrab

def take_screenshot():
    # 等待几秒钟以便用户准备
    time.sleep(3)
    
    # 获取屏幕截图
    screenshot = ImageGrab.grab()
    
    # 保存截图到文件
    screenshot.save(r"C:\Users\34314\Desktop\YOLO\screenshot.png")

if __name__ == "__main__":
    input("按回车键开始截图...")
    take_screenshot()
    print("屏幕截图已保存为 screenshot.png")