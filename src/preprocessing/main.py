import lydorn_utils.geo_utils as utils
import osgeo
import rasterio
from rasterio.windows import Window
import numpy as np
from PIL import Image, ImageDraw

if __name__ == '__main__':
    print(osgeo.gdal.__version__)
    # with rasterio.open('../../data/niedersachsen/large/out.tif') as src:
    # with rasterio.open('../../data/toulouse_no_geo/large/3.tif') as src:
    with rasterio.open('../../data/zeven/large/dop20rgb_32_516_5904_2_ni_2018-03-18.tif') as src:
        window = Window(0, 0, 400, 400)
        kwargs = src.meta.copy()
        kwargs.update({
            'height': window.height,
            'width': window.width,
            'transform': rasterio.windows.transform(window, src.transform)})

        with rasterio.open('cropped.tif', 'w', **kwargs) as dst:
            dst.write(src.read(window=window))

    image_to_show = Image.open('cropped.tif')
    img_draw = ImageDraw.Draw(image_to_show)
    poligons = utils.get_polygons_from_osm("cropped.tif", tag="building", ij_coords=False)
    for p in poligons:
        result = list(map(tuple, np.array(p).astype(int)))
        print(result)
        img_draw.polygon(result, fill='#ee7621', outline='#ee7621')
        image_to_show.save("cropped_poligons.tif")
