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
import numpy as np
import cv2
import json


def get_rotate_crop_image(img, points):
    '''
    img_height, img_width = img.shape[0:2]
    left = int(np.min(points[:, 0]))
    right = int(np.max(points[:, 0]))
    top = int(np.min(points[:, 1]))
    bottom = int(np.max(points[:, 1]))
    img_crop = img[top:bottom, left:right, :].copy()
    points[:, 0] = points[:, 0] - left
    points[:, 1] = points[:, 1] - top
    '''

    img_crop_width = int(
        max(
            np.linalg.norm(points[0] - points[1]),
            np.linalg.norm(points[2] - points[3])))
    img_crop_height = int(
        max(
            np.linalg.norm(points[0] - points[3]),
            np.linalg.norm(points[1] - points[2])))
    pts_std = np.float32([[0, 0], [img_crop_width, 0],
                          [img_crop_width, img_crop_height],
                          [0, img_crop_height]])
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img,
        M, (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC)
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    if dst_img_height * 1.0 / dst_img_width >= 1.5:
        # 将矩阵逆时针旋转90度
        dst_img = np.rot90(dst_img)
    return dst_img


def convert_label_infor(label_infor):
    label_infor = label_infor.decode()
    label_infor = label_infor.encode('utf-8').decode('utf-8-sig')
    substr = label_infor.strip("\n").split("\t")
    img_path = substr[0]
    label = json.loads(substr[1])
    return img_path, label


def det2rec(input_path, output_path, ratio):
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
            output_path + "ocr_rec_train_label.txt", "w", encoding="utf-8")
        eval_file = open(output_path + "ocr_rec_eval_label.txt",
                         "w", encoding="utf-8")
        test_file = open(output_path + "ocr_rec_test_label.txt",
                         "w", encoding="utf-8")
    else:
        train_file = open(
            output_path + "ocr_rec_train_label.txt", "w", encoding="utf-8")
        test_file = open(output_path + "ocr_rec_test_label.txt",
                         "w", encoding="utf-8")

    with open(input_path, "rb") as fin:
        label_infor_list = fin.readlines()
        for i in range(len(label_infor_list)):
            substr, label = convert_label_infor(label_infor_list[i])
            img_path = "train_data/ocr_det/" + substr
            img = cv2.imread(img_path)

            for i, dict in enumerate(label):
                total_num += 1
                points = dict["points"]
                points = np.float32(np.array(points))

                dst_img = get_rotate_crop_image(img, points)
                dst_img_path = "train_data/ocr_rec/" + \
                    substr[0:-4] + "_" + str(i) + ".jpg"
                cv2.imwrite(dst_img_path, dst_img)
                if len(ratio) == 3:
                    if 1 <= total_num % (train_ratio+eval_ratio+test_ratio) <= train_ratio:
                        train_file.write(
                            substr[0:-4] + "_" + str(i) + ".jpg" + '\t' + dict["transcription"] + '\n')

                    if train_ratio < total_num % (train_ratio+eval_ratio+test_ratio) <= train_ratio+eval_ratio:
                        eval_file.write(
                            substr[0:-4] + "_" + str(i) + ".jpg" + '\t' + dict["transcription"] + '\n')

                    else:
                        test_file.write(
                            substr[0:-4] + "_" + str(i) + ".jpg" + '\t' + dict["transcription"] + '\n')
                else:
                    if 1 <= total_num % (train_ratio+test_ratio) <= train_ratio:
                        train_file.write(
                            substr[0:-4] + "_" + str(i) + ".jpg" + '\t' + dict["transcription"] + '\n')
                    else:
                        test_file.write(
                            substr[0:-4] + "_" + str(i) + ".jpg" + '\t' + dict["transcription"] + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input_path',
        type=str,
        default="train_data/ocr_rec_label.txt",
        help='Input path to be converted')
    parser.add_argument(
        '--output_path',
        type=str,
        default="train_data/ocr_rec/",
        help='Output file name')
    parser.add_argument(
        '--ratio',
        type=list,
        default=[3, 1, 1],
        help='train_ratio:eval_ratio:test_ratio or train_ratio : test_ratio')

    args = parser.parse_args()
    det2rec(args.input_path, args.output_path,  args.ratio)
