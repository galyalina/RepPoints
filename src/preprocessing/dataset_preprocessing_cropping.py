import slidingwindow as sw
import cv2
import os

# DIRECTORY_CROPPED_IMAGE = "../../data/train/"
# DIRECTORY_CROPPED_MASK = "../../data/mask/"
# DIRECTORY_IMAGE = "../../data/train_large/"
# DIRECTORY_MASK = "../../data/mask_large/"
DIRECTORY_CROPPED_IMAGE = "../../latest/train/"
DIRECTORY_CROPPED_MASK = "../../latest/mask/"
DIRECTORY_IMAGE = "../../latest/train_large/"
DIRECTORY_MASK = "../../latest/mask_large/"
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


def start_points(size, split_size, overlap=0):
    points = [(0, min(size, split_size))]
    stride = int(split_size * (1 - overlap))
    counter = 1
    while True:
        pt = stride * counter
        if pt + split_size >= size:
            points.append((pt, size-pt))
            break
        else:
            points.append((pt, split_size))
        counter += 1
    return points


def crop_images(original, mask, str_prefix):
    # Generate the set of windows, with a 256-pixel max window size and 50% overlap
    windows = sw.generate(original, ['h', 'w', 'c'], IMAGE_SIZE, IMAGE_OVERLAP_PERCENTAGE)
    height, width, channels = original.shape
    X_points = start_points(width, IMAGE_SIZE, IMAGE_OVERLAP_PERCENTAGE)
    Y_points = start_points(height, IMAGE_SIZE, IMAGE_OVERLAP_PERCENTAGE)

    index = 0
    for y, h in Y_points:
        for x, w in X_points:
            cropped_image = original[y:y + h, x:x + w]
            cropped_mask = mask[y:y + h, x:x + w]
            cv2.imwrite(DIRECTORY_CROPPED_IMAGE + '' + str_prefix + str(index) + '.tif', cropped_image)
            cv2.imwrite(DIRECTORY_CROPPED_MASK + '' + str_prefix + str(index) + '.tif', cropped_mask)
            index += 1

    # for index, single_window in enumerate(windows):
    #     print(single_window)
    #     x = single_window.x
    #     y = single_window.y
    #     width = single_window.w
    #     height = single_window.h
    #     cropped_image = original[y:y + height, x:x + width]
    #     cropped_mask = mask[y:y + height, x:x + width]
    #
    #     cv2.imwrite(DIRECTORY_CROPPED_IMAGE + '' + str_prefix + str(index) + '.tif', cropped_image)
    #     cv2.imwrite(DIRECTORY_CROPPED_MASK + '' + str_prefix + str(index) + '.tif', cropped_mask)


def main():
    delete_folder(DIRECTORY_CROPPED_MASK)
    delete_folder(DIRECTORY_CROPPED_IMAGE)
    for subdir, dirs, files in os.walk(DIRECTORY_IMAGE):
        for file in files:
            # TODO delete, used for test only
            # file = "3.tif"
            if not file.lower().endswith(('.tif', '.jpg', '.jpeg')):
                continue
            print('File number', file.split('.')[0])
            # Load our input image here
            print('File number', DIRECTORY_IMAGE + file)
            print('File number', DIRECTORY_MASK + file)
            image = cv2.imread(DIRECTORY_IMAGE + file, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            mask = cv2.imread(DIRECTORY_MASK + file, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            crop_images(image, mask, file.split('.')[0])
            # TODO delete, used for test only
            # break
    # Calculate number of generated images with masks
    path, dirs, files = next(os.walk(DIRECTORY_CROPPED_MASK))
    file_count = len(files)
    print(f'\n{file_count} images are generated\n')


if __name__ == '__main__':
    main()
