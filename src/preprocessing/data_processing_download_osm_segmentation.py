import os

import lydorn_utils.geo_utils as utils
import numpy as np
import rasterio
from PIL import Image, ImageDraw
from rasterio.windows import Window

PATH = "../../data/test_tiff_cropping"
DIRECTORY_CROPPED_IMAGE = PATH + "/train/"
DIRECTORY_MASK_IMAGE = PATH + "/mask/"


# def generate_new_image(src, image_path):
# with rasterio.open(src) as src:
#     window = Window(0, 0, width=src.width, height=src.height)
#     kwargs = src.meta.copy()
#     kwargs.update({
#         'height': window.height,
#         'width': window.width,
#         'transform': rasterio.windows.transform(window, src.transform)})
#
#     with rasterio.open(image_path, 'w', **kwargs) as dst:
#         dst.write()
#         return dst


def get_segmentation(image_name):
    with rasterio.open(DIRECTORY_CROPPED_IMAGE + image_name) as src:
        mask = Image.new('1', (src.width, src.height), "#000000")
        mask_draw = ImageDraw.Draw(mask)
        poligons = utils.get_polygons_from_osm(DIRECTORY_CROPPED_IMAGE + image_name, tag="building", ij_coords=False)
        for p in poligons:
            result = list(map(tuple, np.array(p).astype(int)))
            mask_draw.polygon(result, fill='#ee7621', outline='#ee7621')
        mask.save(DIRECTORY_MASK_IMAGE + image_name)


if __name__ == '__main__':
    for subdir, dirs, files in os.walk(DIRECTORY_CROPPED_IMAGE):
        for file in files:
            if file.lower().endswith('.tif'):
                if not os.path.exists(DIRECTORY_MASK_IMAGE + file):
                    print("get mask for " + file)
                    get_segmentation(file)
                # else:
                    # print("mask " + file + " already exists")
