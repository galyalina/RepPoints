import os
import numpy as np  # (pip install numpy)
from matplotlib import patches
from matplotlib.patches import Rectangle
from skimage import measure  # (pip install scikit-image)
from shapely.geometry import Polygon, MultiPolygon  # (pip install Shapely)
import json
from PIL import Image  # (pip install Pillow)
import matplotlib.pyplot as plt
import cv2
from skimage.measure import label, regionprops

DIRECTORY_ANNOTATIONS = "../../data_test/annotations/"
DIRECTORY_CROPPED_IMAGE = "../../data_test/train/"
DIRECTORY_CROPPED_MASK = "../../data_test/mask/"
DIRECTORY_SPLIT_MASK = "../../data_test/mask_split/"

impervious_surface_id, building_id, previous_surface_id, high_vegetation_id, car_id, water_id, sport_venues_id, void_id = [
    1, 2, 3, 4, 5, 6, 7, 8]

category_name = ['impervious_surface', 'building', 'previous_surface', 'high_vegetation', 'car', 'water',
                 'sport_venues',
                 'void']

category_ids = {
    '(38, 38, 38)': impervious_surface_id,
    '(238, 118, 33)': building_id,
    '(34, 139, 34)': previous_surface_id,
    '(0, 222, 137)': high_vegetation_id,
    '(255, 0, 0)': car_id,
    '(0, 0, 238)': water_id,
    '(160, 30, 230)': sport_venues_id,
    '(255, 255, 255)': void_id
}

colors = {
    '(38, 38, 38)': "#262626",
    '(238, 118, 33)': '#ee7621',
    '(34, 139, 34)': '#228b22',
    '(0, 222, 137)': "#00de89",
    '(255, 0, 0)': "#ff0000",
    '(0, 0, 238)': "#0000ee",
    '(160, 30, 230)': "#a01ee6",
    '(255, 255, 255)': "#ffffff"
}


# impervious surface & 38, 38, 38 & dark grey
# building & 238, 118, 33 & orange
# previous surface &  34, 139, 34 & dark green
# high vegetation &  0, 222, 137 & bright green
# car &  255, 0, 0 & red
# water &  0, 0, 238 & blue
# sport venues & 160, 30, 230 & purple
# void &  255, 255, 255 &

def create_sub_mask_annotation(sub_mask, image_id, category_id, annotation_id, is_crowd):
    annotations = []
    # Find contours (boundary lines) around each sub-mask
    # Note: there could be multiple contours if the object
    # is partially occluded. (E.g. an elephant behind a tree)
    # print(sub_mask.shape())
    img_array = np.array(sub_mask)

    bboxes = get_bbox(img_array)
    for box in bboxes:
        annotation = {
            'iscrowd': is_crowd,
            'image_id': image_id,
            'category_id': category_id,
            'id': annotation_id,
            'bbox': box,
            'area': box[3] * box[2]
        }
        annotation_id = annotation_id+1
        annotations.append(annotation)
    annotation_id = annotation_id + 1
    return annotation_id, annotations


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


# TODO remove, this function creates submasks from given dir and stores in another
# def store_sub_masks():
#     for subdir, dirs, files in os.walk(DIRECTORY_CROPPED_MASK):
#         for file in files:
#             # # TODO delete, used for test only
#             # file = "3.tif"
#             if not file.lower().endswith(('.tiff', '.tif', '.jpg', '.jpeg')):
#                 continue
#             with Image.open(DIRECTORY_CROPPED_MASK + file) as image:
#                 submasks = create_sub_masks(image)
#                 print(type(submasks))
#                 print(submasks)
#                 index = 0
#                 for key, value in submasks.items():
#                     print(key, ' : ', type(value), ' : ', value, '\n')
#                     value.save(DIRECTORY_SPLIT_MASK + 'img_' + file.split('.')[0] + str(index) + ".jpeg", "JPEG")
#                     index = index + 1
#                 # for submask in submasks:
#                 #     submask.save(DIRECTORY_SPLIT_MASK + 'img_' + file.split('.')[0] + str(index) + ".jpeg", "JPEG")
#             # break
#     # Calculate number of generated images with masks
#     path, dirs, files = next(os.walk(DIRECTORY_CROPPED_MASK))
#     file_count = len(files)
#     print(f'\n{file_count} images are generated\n')


def generate_annotation_for_single_image(annotations, file, annotation_id_index, image_id_index, is_crowd_flag=False):
    # store_sub_masks()
    try:
        mask_image = Image.open(DIRECTORY_CROPPED_MASK + file)
        original_image = Image.open(DIRECTORY_CROPPED_IMAGE + file)
    except FileNotFoundError:
        raise Exception("Your dataset is corrupted, check ${file}")
    sub_masks = create_sub_masks(mask_image)
    fig, ax = plt.subplots()
    ax.imshow(original_image)
    # plt.show()
    for color, sub_mask in sub_masks.items():
        # if color != '(238, 118, 33)':
        #     continue
        category_id = category_ids[color]
        annotation_id, category_annotations = create_sub_mask_annotation(sub_mask, image_id_index,
                                                                         category_id,
                                                                         annotation_id_index,
                                                                         is_crowd_flag)
        annotations.extend(category_annotations)
        annotation_id_index += 1

        for single_annotation in category_annotations:
            x, y, w, h = single_annotation['bbox']
            ax.add_patch(
                Rectangle((x, y), w, h, linewidth=1,
                          edgecolor=colors[color],
                          facecolor='none'))

    plt.show()
    return annotations, annotation_id_index
    # break


# todo
def main():
    annotation_id_index = 1
    annotations = []
    images = []
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
    categories = [{"id": index, "name": name, "supercategory": name}
                  for index, name in enumerate(category_name, start=1)]

    # only when one category building exists
    # categories = [{"id": 2, "name": 'building', "supercategory": 'building'}]

    coco = {
        'info': info,
        'licenses': licenses,
        'type': 'instances',
        'images': images,
        'annotations': annotations,
        'categories': categories
    }
    with open(DIRECTORY_ANNOTATIONS + 'coco.json', 'w') as outfile:
        json.dump(coco, outfile)


if __name__ == '__main__':
    main()
