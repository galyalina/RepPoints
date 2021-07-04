import os
import numpy as np  # (pip install numpy)
from skimage import measure  # (pip install scikit-image)
from shapely.geometry import Polygon, MultiPolygon  # (pip install Shapely)
import json
from PIL import Image  # (pip install Pillow)

DIRECTORY_ANNOTATIONS = "../../data/annotations/"
DIRECTORY_CROPPED_IMAGE = "../../data/train/"
DIRECTORY_CROPPED_MASK = "../../data/mask/"
DIRECTORY_IMAGE = "../../data/train_large/"
DIRECTORY_MASK = "../../data/mask_large/"
DIRECTORY_SPLIT_MASK = "../../data/mask_split/"

impervious_surface_id, building_id, previous_surface_id, high_vegetation_id, car_id, water_id, sport_venues_id, void_id = [
    1, 2, 3, 4, 5, 6, 7, 8]

category_ids = {
    1: {
        '(38, 38, 38)': impervious_surface_id,
        '(238, 118, 33)': building_id,
        '(34, 139, 34)': previous_surface_id,
        '(0, 222, 137)': high_vegetation_id,
        '(255, 0, 0)': car_id,
        '(0, 0, 238)': water_id,
        '(160, 30, 230)': sport_venues_id,
        '(255, 255, 255)': void_id
    }
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
    # Find contours (boundary lines) around each sub-mask
    # Note: there could be multiple contours if the object
    # is partially occluded. (E.g. an elephant behind a tree)
    # print(sub_mask.shape())
    img_array = np.array(sub_mask)
    contours = measure.find_contours(img_array, 0.5, positive_orientation='low')

    segmentations = []
    polygons = []
    for contour in contours:
        # Flip from (row, col) representation to (x, y)
        # and subtract the padding pixel
        for i in range(len(contour)):
            row, col = contour[i]
            contour[i] = (col - 1, row - 1)

        # Make a polygon and simplify it
        poly = Polygon(contour)
        poly = poly.simplify(1.0, preserve_topology=False)
        polygons.append(poly)
        segmentation = np.array(poly.exterior.coords).ravel().tolist()
        segmentations.append(segmentation)

    # Combine the polygons to calculate the bounding box and area
    multi_poly = MultiPolygon(polygons)
    x, y, max_x, max_y = multi_poly.bounds
    width = max_x - x
    height = max_y - y
    bbox = (x, y, width, height)
    area = multi_poly.area

    annotation = {
        'segmentation': segmentations,
        'iscrowd': is_crowd,
        'image_id': image_id,
        'category_id': category_id,
        'id': annotation_id,
        'bbox': bbox,
        'area': area
    }

    return annotation


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
                    sub_masks[pixel_str] = Image.new('1', (width + 2, height + 2))

                # Set the pixel value to 1 (default is 0), accounting for padding
                sub_masks[pixel_str].putpixel((x + 1, y + 1), 1)
    return sub_masks


# TODO remove, this function creates submasks from given dir and stores in another
# def create_sub_masks():
#     for subdir, dirs, files in os.walk(DIRECTORY_MASK):
#         for file in files:
#             # TODO delete, used for test only
#             file = "3.tif"
#             if not file.lower().endswith(('.tiff', '.tif', '.jpg', '.jpeg')):
#                 continue
#             with Image.open(DIRECTORY_MASK + file) as image:
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
#             break
#     # Calculate number of generated images with masks
#     path, dirs, files = next(os.walk(DIRECTORY_CROPPED_MASK))
#     file_count = len(files)
#     print(f'\n{file_count} images are generated\n')


# todo
def main(is_crowd_flag, annotation_id_index, image_id_index):
    for subdir, dirs, files in os.walk(DIRECTORY_CROPPED_MASK):
        for file in files:
            # TODO delete, used for test only
            file = "test_mask.tiff"
            if not file.lower().endswith(('.tiff', '.tif', '.jpg', '.jpeg')):
                continue
            mask_image = Image.open(DIRECTORY_CROPPED_MASK + file)
            sub_masks = create_sub_masks(mask_image)
            for color, sub_mask in sub_masks.items():
                category_id = category_ids[image_id_index][color]
                annotation = create_sub_mask_annotation(sub_mask, image_id_index, category_id, annotation_id_index,
                                                        is_crowd_flag)
                annotations.append(annotation)
                annotation_id_index += 1
        image_id_index += 1
        break

    print(json.dumps(annotations))
    with open(DIRECTORY_ANNOTATIONS+'annotations.json', 'w') as outfile:
        json.dump(annotations, outfile)


if __name__ == '__main__':
    is_crowd = 0
    # These ids will be automatically increased as we go
    annotation_id = 1
    image_id = 1
    # Create the annotations
    annotations = []
    main(is_crowd, annotation_id, image_id)
