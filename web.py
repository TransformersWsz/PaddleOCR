from flask import Flask
from flask import jsonify
from flask import render_template
from flask import request
from flask import redirect
from werkzeug.utils import secure_filename
import paddlehub as hub
import json

import argparse
from datetime import timedelta
from tools.infer.predict_system import init_ocr_model
from tools.infer.predict_system import ft_recognize
from extract import init_model
from extract import recognize
from extract import filter_info


import json
app = Flask(__name__)
# app.config["SEND_FILE_MAX_AGE_DEFAULT"] = timedelta(0)

upload_config_path = "./newrule.json"    # 规则配置文件上传存放位置
config_json_path = "./bill_config.json"

img_path = "./upload/origin.jpg"
use_gpu = False
det_model_dir = "./output/inference/db_det/"
rec_model_dir = "./output/inference/rec_CRNN"

text_sys, args = init_ocr_model(img_path, use_gpu, det_model_dir, rec_model_dir)


@app.route('/')
def hello(name=None):
    with open(config_json_path, "r", encoding="utf8") as fr:
        config = json.load(fr)
    return render_template('hello.html', name=name, config_names=config.keys())


@app.route('/rule')
def rule(name=None):
    return render_template('rule.html')


def update_config(config_json_path, upload_config_path):
    with open(config_json_path, "r", encoding="utf8") as fr:
        config = json.load(fr)
    merge_res = True
    try:
        with open(upload_config_path, "r", encoding="utf8") as fr:
            new_rule = json.load(fr)
            for k, v in new_rule.items():
                config[k] = v
            with open(config_json_path, "w", encoding="utf8") as fw:
                json.dump(config, fw, ensure_ascii=False)
    except Exception:
        print("json格式非法")
        res = False
    return merge_res



@app.route('/addrule', methods=['POST'])
def addrule():
    f = request.files['jsonfile']
    f.save(upload_config_path)
    merge_res = update_config(config_json_path, upload_config_path)
    res = "success" if merge_res == True else "json格式非法"
    return jsonify({
        "msg": res
    })


@app.route('/upload', methods=['POST'])
def upload_file():
    
    f = request.files['picfile']
    f.save(img_path)
    bill_name = request.values.get("bill_name")

    return jsonify({
        "msg": "success"
    })


@app.route("/result")
def result():
    bill_name = request.args.get("bill_name")
    data, save_path = ft_recognize(text_sys, args)
    shop_name, order_number, pay_money, sell_time = filter_info(data, bill_name)
    print(shop_name, order_number, pay_money, sell_time)

    idx = save_path.find("/static/")
    rec_result_path = save_path[idx:]
    return render_template("result.html", img_path=rec_result_path, tgt_info=(shop_name, order_number, sell_time, pay_money), enum=enumerate(data))


if __name__ == "__main__":
    app.run(host="172.16.0.11", debug=True)
