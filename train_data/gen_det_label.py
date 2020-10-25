# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import argparse
import json
import os


def gen_det(input_path, output_path, ratio):
    assert len(ratio) == 2 or len(ratio) == 3, "len(ratio) must be 2 or 3"
    for num in ratio:
        assert type(num) == int, "ratio's num must be int"
    if len(ratio) == 2:
        train_ratio, test_ratio = ratio
    else:
        train_ratio, eval_ratio, test_ratio = ratio
    total_num = 0
    if len(ratio) == 3:
        train_file = open(
            output_path+"ocr_rec_train_label.txt", "w", encoding="utf-8")
        eval_file = open(output_path+"ocr_rec_eval_label.txt",
                         "w", encoding="utf-8")
        test_file = open(output_path+"ocr_rec_test_label.txt",
                         "w", encoding="utf-8")
    else:
        train_file = open(
            output_path + "ocr_rec_train_label.txt", "w", encoding="utf-8")
        test_file = open(output_path + "ocr_rec_test_label.txt",
                         "w", encoding="utf-8")
    for root, dirs, files in os.walk(input_path):
        for name in files:
            with open(os.path.join(root, name), "r", encoding="utf-8") as fp:
                json_data = json.load(fp)
                total_num += 1
                label = []
                for poly in json_data["shapes"]:

                    poly_label = {}
                    poly_label["transcription"] = poly["label"]
                    new_points = []
                    for point in poly["points"]:
                        new_points.append([int(point[0])+1, int(point[1])+1])
                    final_points = []
                    for i in range(len(new_points)):
                        final_points.append(list(new_points[i]))
                        final_points.append(
                            [new_points[(i+1) % 2][0], new_points[i][1]])

                    poly_label["points"] = final_points
                    label.append(poly_label)
                if len(ratio) == 3:
                    if 1 <= total_num % (train_ratio+eval_ratio+test_ratio) <= train_ratio:
                        train_file.write(
                            "ocr_images/"+name.split(".")[0]+".jpg" + '\t' + json.dumps(label, ensure_ascii=False) + '\n')
                    elif train_ratio < total_num % (train_ratio+eval_ratio+test_ratio) <= train_ratio+eval_ratio:
                        eval_file.write(
                            "ocr_images/" + name.split(".")[0] + ".jpg" + '\t' + json.dumps(label, ensure_ascii=False) + '\n')
                    else:
                        test_file.write(
                            "ocr_images/" + name.split(".")[0] + ".jpg" + '\t' + json.dumps(label, ensure_ascii=False) + '\n')
                else:
                    if 1 <= total_num % (train_ratio+test_ratio) <= train_ratio:
                        train_file.write(
                            "ocr_images/"+name.split(".")[0]+".jpg" + '\t' + json.dumps(label, ensure_ascii=False) + '\n')
                    else:
                        test_file.write(
                            "ocr_images/" + name.split(".")[0] + ".jpg" + '\t' + json.dumps(label, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input_path',
        type=str,
        default="train_data/json",
        help='Input path to be converted')
    parser.add_argument(
        '--output_path',
        type=str,
        default="train_data/ocr_det/",
        help='Output file name')
    parser.add_argument(
        '--ratio',
        type=list,
        default=[3, 1, 1],
        help='train_ratio:eval_ratio:test_ratio or train_ratio : test_ratio')

    args = parser.parse_args()
    gen_det(args.input_path, args.output_path, args.ratio)
