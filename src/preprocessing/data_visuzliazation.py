import json
import bbox_visualizer as bbv
import cv2

DIRECTORY_IMAGE = "../../data_test/train/"
ANNOTATIONS = "../../data_test/annotations/coco.json"


def main():
    img = cv2.imread(DIRECTORY_IMAGE + 'img_img_853.tif')
    annotations = json.load(open(ANNOTATIONS))['annotations']
    for i in annotations:
        [x, y, w, h] = i['bbox']
        p1_x = int(x)
        p2_x = int(x + w)
        p1_y = int(y)
        p2_y = int(y + h)

        cv2.rectangle(img, (p1_x, p1_y), (p2_x, p2_y), (255, 0, 0), 1)
        cv2.imshow(' ', img)
        cv2.waitKey(0)


if __name__ == '__main__':
    main()
