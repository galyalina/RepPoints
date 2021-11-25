import rasterio
import os

from rasterio.windows import Window

PATH = "../../data/test_tiff_cropping"
DIRECTORY_CROPPED_IMAGE = PATH + "/train/"
DIRECTORY_IMAGE = PATH + "/large/"
IMAGE_SIZE = 460
IMAGE_OVERLAP_PERCENTAGE = 0.5


def find_file(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)


def delete_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            try:
                os.remove(os.path.join(root, name))
            except Exception:
                print()
        print()
        for name in dirs:
            try:
                os.rmdir(os.path.join(root, name))
            except Exception:
                print()


def start_points(size, split_size, overlap=0.0):
    points = [(0, min(size, split_size))]
    stride = int(split_size * (1 - overlap))
    counter = 1
    while True:
        pt = stride * counter
        if pt + split_size >= size:
            points.append((pt, size - pt))
            break
        else:
            points.append((pt, split_size))
        counter += 1
    return points


def crop_images(original, str_prefix):
    with rasterio.open(original) as src:
        height = src.height
        width = src.width
        X_points = start_points(width, IMAGE_SIZE, IMAGE_OVERLAP_PERCENTAGE)
        Y_points = start_points(height, IMAGE_SIZE, IMAGE_OVERLAP_PERCENTAGE)
        index = 0
        for y, h in Y_points:
            for x, w in X_points:
                with rasterio.open(original) as src:
                    window = Window(x, y, w, h)
                    kwargs = src.meta.copy()
                    kwargs.update({
                        'height': window.height,
                        'width': window.width,
                        'transform': rasterio.windows.transform(window, src.transform)})
                    with rasterio.open(DIRECTORY_CROPPED_IMAGE + '' + str_prefix + str(index) + '.tif', 'w',
                                       **kwargs) as dst:
                        dst.write(src.read(window=window))
                        index += 1
        return index


def main():
    delete_folder(DIRECTORY_CROPPED_IMAGE)
    for subdir, dirs, files in os.walk(DIRECTORY_IMAGE):
        for file in files:
            if not file.lower().endswith('.tif'):
                continue
            print('File number', file.split('.')[0])
            # Load our input image here
            print('File number', DIRECTORY_IMAGE + file)
            number_of_cropped_images = crop_images(DIRECTORY_IMAGE + file, file.split('.')[0])
            # TODO delete, used for test only
            print(f'\n{number_of_cropped_images} images are generated for {file}\n')
    # Calculate number of generated images with masks


if __name__ == '__main__':
    main()
