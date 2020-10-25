"""
提取目标文字，例如店铺名称、销售时间、流水单号、实付金额
"""


import paddlehub as hub
import os
import cv2
import time
import re
from billextract import BillExtract


def format_time(sell_time):
    """格式化输出时间"""
    # date = sell_time[:10]
    # hms = sell_time[10:].strip()
    # hour, minute, second = re.findall("\d+", hms)
    # hour = hour.strip()
    # minute = minute.strip()
    # second = second.strip()
    # return "{} {}:{}:{}".format(date, hour, minute, second)

    date = sell_time[:10]
    hms = sell_time[10:].strip()

    return "{} {}".format(date, hms)


def is_contained(key_list, text):
    """判断关键字是否在text中"""
    for item in key_list:
        if item in text:
            return True
    return False


def find_shop_name(data):
    shop_name = "未检测出店铺名称，请调整拍摄角度"
    name_keywords = ["店", "超市", "商场"]
    is_name_checked = False
    for idx, item in enumerate(data):
        try:
            if is_name_checked == False and is_contained(name_keywords, item[0]):
                shop_name = item[0]
                is_name_checked = True
                break
        except Exception:
            pass
    return shop_name


def find_order_number(data):
    order_number = "未检测出订单号，请调整拍摄角度"
    order_keywords = ["票号", "流水", "单号"]
    is_order_checked = False
    order_pattern = "\d+"
    for idx, item in enumerate(data):
        try:
            if is_order_checked == False and is_contained(order_keywords, item[0]):
                res = re.findall(order_pattern, item[0])
                order_number = res[0] if len(res) > 0 else re.findall(
                    order_pattern, data[idx+1][0])[0]
                is_order_checked = True
        except Exception:
            pass
    return order_number


def match_time_text(time_patterns, text):
    res = None
    for p in time_patterns:
        t = re.search(p, text)
        if t != None:
            start_idx = t.span()[0]
            res = text[start_idx:].strip()
            break
    return res


def find_sell_time(data):
    sell_time = "未检测出销售时间，请调整拍摄角度"
    time_keywords = ["时间"]
    is_time_checked = False
    time_patterns = ["\d{4}-\d{2}-\d{2}"]
    for idx, item in enumerate(data):
        try:
            match_time = match_time_text(time_patterns, item[0])
            if match_time:
                sell_time = format_time(match_time)
        except Exception:
            pass
    return sell_time


def get_keyword_idx(pay_keywords, text):
    idx = -1
    for word in pay_keywords:
        if word in text:
            idx = text.find(word)
            break
    return idx


def get_pay_money_from_candidates(candidate_money):
    d = {}
    for item in candidate_money:
        d[item] = 1 if item not in d else d[item]+1
    l = sorted(d.items(), key=lambda item: item[1], reverse=True)
    return l[0][0]


def find_pay_money(data):
    pay_money = "未检测出实付金额，请调整拍摄角度"
    pay_keywords = ["应收", "支付", "应付", "合计", "总数", "现金", "微信", "支付宝", "已付"]
    is_pay_checked = False
    money_pattern = "\d+\.?\d*"
    candidate_money = []
    for idx, item in enumerate(data):
        try:
            if is_pay_checked == False and is_contained(pay_keywords, item[0]):
                res = re.findall(money_pattern, item[0])
                # 如果检测出的文字是”实付金额  100元“，那么可以直接提取；
                # 如果检测出的文字是”实付金额“，那么通常下一个text即为金额数字
                if len(res) > 0:
                    keyword_idx = get_keyword_idx(pay_keywords, item[0])
                    money_idx = item[0].find(res[0])
                    if money_idx > keyword_idx:    # 模型可能会误识别成"12应收"
                        candidate_money.append(res[0])
                else:
                    tmp = re.findall(money_pattern, data[idx+1][0])[0]
                    candidate_money.append(tmp)
                # is_pay_checked = True
        except Exception:
            pass
    try:
        pay_money = get_pay_money_from_candidates(candidate_money)
    except Exception:
        pass
    
    return pay_money


def filter_info(data, bill_name="common"):
    """提取出四个目标信息"""
    if bill_name != "common":
        be = BillExtract(bill_name)
        shop_name, order_number, pay_money, sell_time = be.extract(data)
    else:
        shop_name = find_shop_name(data)
        order_number = find_order_number(data)
        pay_money = find_pay_money(data)
        sell_time = find_sell_time(data)
    return shop_name, order_number, pay_money, sell_time


def init_model():
    ocr = hub.Module(name="chinese_ocr_db_crnn_mobile")
    return ocr


def recognize(ocr, img_path, output_dir="/home/antcoder/ocr_dist/static/img"):
    start_time = time.time()
    result = ocr.recognize_text(images=[cv2.imread(img_path)],
                                visualization=True, output_dir=output_dir)
    save_path, data = result[0]["save_path"], result[0]["data"]
    shop_name, order_number, pay_money, sell_time = filter_info(data)
    end_time = time.time()
    print("consumed time:{:.2f}s".format(end_time-start_time))
    print(shop_name, order_number, pay_money, sell_time)

    return save_path, data, shop_name, order_number, pay_money, sell_time


def batch_recognize(img_dir, output_dir):
    """
    img_dir: 待识别的图片目录
    output_dir: 输出目录，每张原图片(img_dir/test.jpg)对应两个输出文件：
        1：模型标注后的图片(output_dir/test.jpg)
        2：模型从图片中识别出的文字(output_dir/test.txt)
    """
    ocr = init_model()
    files = os.listdir(img_dir)
    for filename in files:
        if filename.endswith(".jpg"):
            img_path = os.path.join(img_dir, filename)
            save_path, data, shop_name, order_number, pay_money, sell_time = recognize(
                ocr, img_path, output_dir=output_dir)
            tgt_img_path = os.path.join(output_dir, filename)
            tgt_txt_path = os.path.join(output_dir, filename[:-3]+"txt")
            os.rename(save_path, tgt_img_path)
            with open(tgt_txt_path, "w", encoding="utf8") as fw:
                fw.write("店铺名称\t流水单号\t销售时间\t实付金额\n")
                fw.write("{}\t{}\t{}\t{}\n\n".format(
                    shop_name, order_number, sell_time, pay_money))

                fw.write("序号\t名称\t置信度\n")
                for idx, item in enumerate(data):
                    fw.write("{}\t{}\t{}\n".format(
                        idx, item["text"], item["confidence"]))
            print(
                "====================={} 识别结束！========================".format(img_path))


def eval():
    ocr = init_model()
    result = ocr.recognize_text(images=[cv2.imread("/home/antcoder/ocr_dist/upload/origin.jpg")],
                                visualization=True, output_dir="/home/antcoder/ocr_dist/output")
    print(result)


if __name__ == "__main__":
    batch_recognize("/home/antcoder/ocr_dist/upload",
                    "/home/antcoder/ocr_dist/output")

    # ocr = init_model()
    # recognize(ocr, "/home/antcoder/ocr_dist/pics/jpg_wei/2020-07-21 180816.jpg")
