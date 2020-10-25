import json
import re

class BillExtract(object):
    """根据指定规则去抽取小票目标文字"""
    def __init__(self, name, config_path="./bill_config.json"):
        with open(config_path, "r", encoding="utf8") as fr:
            config = json.load(fr)
            specific_config = config[name]
            self._shop = specific_config["shop"]
            self._pdate = specific_config["date"]
            self._phms = specific_config["hms"]
            self._order_keyword = specific_config["order_keyword"]
            self._porder = specific_config["order"]
            self._money_keyword = specific_config["money_keyword"]
            self._money_offset = specific_config["money_offset"]
            self._pmoney_num = specific_config["money_num"]


    def extract(self, data):
        shop_name = "未检测出店铺名称，请调整拍摄角度"
        order_number = "未检测出订单号，请调整拍摄角度"
        sell_time = "未检测出销售时间，请调整拍摄角度"
        pay_money = "未检测出实付金额，请调整拍摄角度"
        for idx, item in enumerate(data):
            if self._shop in item[0]:
                shop_name = item[0]
            elif self._order_keyword in item[0]:
                order_number = re.findall(self._porder, item[0])[0]
            elif re.search(self._pdate, item[0]):
                date = re.findall(self._pdate, item[0])[0]
                hms = ""
                try:
                    hms = re.findall(self._phms, item[0])[0]
                except Exception:
                    pass
                finally:
                    sell_time = "{} {}".format(date, hms).strip()
            elif self._money_keyword in item[0]:
                money_text = data[idx+self._money_offset][0]
                pay_money = re.findall(self._pmoney_num, money_text)[0]
        return shop_name, order_number, pay_money, sell_time
