import os
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import json
from PIL import Image, ImageDraw

from skimage.measure import label, regionprops

# We're interested only in buildings
categories = [{"id": 1, "name": 'building', "supercategory": 'none'}]

info = {"year": 2020,
        "version": "1.0",
        "description": "SemCity Toulouse: A benchmark for building instance segmentation in satellite images",
        "contributor": "Roscher, Ribana and Volpi, Michele and Mallet, Clément and Drees, Lukas and Wegner, Jan",
        "url": "http://rs.ipb.uni-bonn.de/data/semcity-toulouse/",
        "date_created": "2020"
        }

licenses = [{"id": 1,
             "name": "Creative Commons Attribution-NonCommercial-ShareAlike 3.0 License",
             "url": "https://creativecommons.org/licenses/by-nc-sa/3.0/"
             }]

DIRECTORY_ANNOTATIONS = "../../data_test/annotations/"
DIRECTORY_CROPPED_IMAGE = "../../data_test/train/"
DIRECTORY_CROPPED_MASK = "../../data_test/mask/"
DIRECTORY_SPLIT_MASK = "../../data_test/mask_split/"

category_name = 'building'
building_id = '(238, 118, 33)'
building_color = '#ee7621'


# impervious surface & 38, 38, 38 & dark grey
# building & 238, 118, 33 & orange
# previous surface &  34, 139, 34 & dark green
# high vegetation &  0, 222, 137 & bright green
# car &  255, 0, 0 & red
# water &  0, 0, 238 & blue
# sport venues & 160, 30, 230 & purple
# void &  255, 255, 255 &

def get_bbox(image):
    # image = np.array(mask_image)
    # idx = image[:, :] > 124
    # image[idx] = 255
    # idx = image[:, :] <= 124
    # image[idx] = 0
    label_img = label(image)
    regions = regionprops(label_img)
    # fig, ax = plt.subplots()
    # ax.imshow(image, cmap=plt.cm.gray)
    # plt.show()
    array_of_boxes = []
    for props in regions:
        min_y, min_x, max_y, max_x = props.bbox
        array_of_boxes.append([min_x, min_y, max_x - min_x, max_y - min_y])
    return array_of_boxes


def create_sub_mask_annotation(sub_mask, image_id, category_id, annotation_id, is_crowd):
    annotations = []
    # Find contours (boundary lines) around each sub-mask
    # Note: there could be multiple contours if the object
    # is partially occluded. (E.g. an elephant behind a tree)
    # print(sub_mask.shape())
    img_array = np.array(sub_mask)

    bboxes = get_bbox(img_array)
    for box in bboxes:
        annotation_id = annotation_id + 1
        annotation = {
            'iscrowd': is_crowd,
            'image_id': image_id,
            'category_id': category_id,
            'id': annotation_id,
            'segmentation': [],
            'bbox': box,
            'area': box[3] * box[2]
        }
        annotations.append(annotation)
    return annotation_id, annotations


def create_sub_masks(mask_image):
    width, height = mask_image.size

    # Initialize a dictionary of sub-masks indexed by RGB colors
    sub_masks = {}
    for x in range(width):
        for y in range(height):
            # Get the RGB values of the pixel
            pixel = mask_image.getpixel((x, y))[:3]

            # If the pixel is not black...
            if pixel != (0, 0, 0):
                # Check to see if we've created a sub-mask...
                pixel_str = str(pixel)
                sub_mask = sub_masks.get(pixel_str)
                if sub_mask is None:
                    # Create a sub-mask (one bit per pixel) and add to the dictionary
                    # Note: we add 1 pixel of padding in each direction
                    # because the contours module doesn't handle cases
                    # where pixels bleed to the edge of the image
                    sub_masks[pixel_str] = Image.new('1', (width, height))

                # Set the pixel value to 1 (default is 0), accounting for padding
                sub_masks[pixel_str].putpixel((x, y), 1)
    return sub_masks


def generate_annotation_for_single_image(annotations, file, annotation_id_index, image_id_index, is_crowd_flag=False):
    try:
        mask_image = Image.open(DIRECTORY_CROPPED_MASK + file)
    except FileNotFoundError:
        raise Exception("Your dataset is corrupted, check ${file}")
    sub_masks = create_sub_masks(mask_image)
    for color, sub_mask in sub_masks.items():
        # we care only for buildings, but if we're not, this line can be uncommented and used for all masks
        if color != '(238, 118, 33)':
            continue
        category_id = 1
        annotation_id_index, category_annotations = create_sub_mask_annotation(sub_mask,
                                                                               image_id_index,
                                                                               category_id,
                                                                               annotation_id_index,
                                                                               is_crowd_flag)
        annotations.extend(category_annotations)
    return annotations, annotation_id_index


# todo
def get_coco_annotations():
    annotations = []
    images = []
    annotation_id_index = 0

    for subdir, dirs, files in os.walk(DIRECTORY_CROPPED_IMAGE):
        for index, file in enumerate(files, start=1):
            mask_image = Image.open(DIRECTORY_CROPPED_IMAGE + file)
            image = {
                'license': 1,
                'file_name': file,
                'height': mask_image.height,
                'width': mask_image.width,
                'id': index
            }
            images.append(image)

            annotations, annotation_id_index = generate_annotation_for_single_image(annotations,
                                                                                    file,
                                                                                    annotation_id_index,
                                                                                    index)
    with open(DIRECTORY_ANNOTATIONS + 'annotations.json', 'w') as outfile:
        json.dump(annotations, outfile)
    with open(DIRECTORY_ANNOTATIONS + 'images.json', 'w') as outfile:
        json.dump(images, outfile)

    coco = {
        'info': info,
        'licenses': licenses,
        'type': 'instances',
        'images': images,
        'annotations': annotations,
        'categories': categories
    }

    return coco


def show_images_with_bbox(coco):
    images = coco['images']
    annotations = coco['annotations']
    ax_dict = dict()
    for image in images:
        fig, ax = plt.subplots()
        ax_dict[image['id']] = ax
        image = Image.open(DIRECTORY_CROPPED_IMAGE + image['file_name'])
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


def store_images_with_bbox(coco):
    images = coco['images']
    annotations = coco['annotations']
    image_dict = dict()
    image_annotations = dict()
    for image in images:
        image_dict[image['id']] = DIRECTORY_CROPPED_IMAGE + image['file_name']
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
        image_to_show.show()


if __name__ == '__main__':
    coco = get_coco_annotations()
    with open(DIRECTORY_ANNOTATIONS + 'coco.json', 'w') as outfile:
        json.dump(coco, outfile)
    # show_images_with_bbox(coco)
    store_images_with_bbox(coco)
