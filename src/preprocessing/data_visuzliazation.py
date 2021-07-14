import json

from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from src.preprocessing.data_annotations_generation_from_buildings_mask import generate_coco_annotations

DIRECTORY_ANNOTATIONS = "../../data_B/annotations/"
DIRECTORY_IMAGE = "../../data_B/train/"
DIRECTORY_MASK = "../../data_B/mask/"
DIRECTORY_VISUALIZATION = "../../data_B/visualization/"

# DIRECTORY_ANNOTATIONS = "../../data/annotations/"
# DIRECTORY_IMAGE = "../../data/train/"
# DIRECTORY_MASK = "../../data/mask/"
# DIRECTORY_VISUALIZATION = "../../data/visualization/"

building_color = '#ee7621'


def show_images_with_bbox(coco):
    images = coco['images']
    annotations = coco['annotations']
    ax_dict = dict()
    for image in images:
        fig, ax = plt.subplots()
        ax_dict[image['id']] = ax
        image = Image.open(DIRECTORY_IMAGE + image['file_name'])
        ax.imshow(image)
    for annotation in annotations:
        image_id = annotation['image_id']
        x, y, w, h = annotation['bbox']
        ax_from_image = ax_dict[image_id]
        ax_from_image.add_patch(Rectangle((x, y), w, h,
                                          linewidth=1,
                                          edgecolor=building_color,
                                          facecolor='none'))
    plt.show()


def store_images_with_bbox(coco, folder):
    images = coco['images']
    annotations = coco['annotations']
    image_dict = dict()
    image_annotations = dict()
    image_name = dict()
    for image in images:
        image_dict[image['id']] = DIRECTORY_IMAGE + image['file_name']
        image_name[image['id']] = image['file_name']
        image_annotations[image['id']] = []
    for annotation in annotations:
        image_id = annotation['image_id']
        x, y, w, h = annotation['bbox']
        (image_annotations[image_id]).append([x, y, w, h])
    for image in image_dict:
        image_path = image_dict[image]
        image_to_show = Image.open(image_path)
        img_draw = ImageDraw.Draw(image_to_show)
        for annotation in image_annotations[image]:
            [x, y, w, h] = annotation
            img_draw.rectangle([(x, y), (x + w, y + h)], outline=building_color, width=3)
        image_to_show.save(folder + image_name[image])
        # image_to_show.show()


if __name__ == '__main__':
    generate_coco_annotations(DIRECTORY_IMAGE, DIRECTORY_MASK, DIRECTORY_ANNOTATIONS)
    file = open(DIRECTORY_ANNOTATIONS + "coco.json", )
    coco = json.load(file)
    # show_images_with_bbox(coco)
    store_images_with_bbox(coco, DIRECTORY_VISUALIZATION)
