import slidingwindow as sw
import cv2
import os

DIRECTORY_CROPPED_IMAGE = "../../data/train/"
DIRECTORY_CROPPED_MASK = "../../data/mask/"
DIRECTORY_IMAGE = "../../data/train_large/"
DIRECTORY_MASK = "../../data/mask_large/"


def find_file(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)


def delete_folder(path):
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
    for name in dirs:
        os.rmdir(os.path.join(root, name))


def crop_images(original, mask, str_prefix):
    # Generate the set of windows, with a 256-pixel max window size and 50% overlap
    windows = sw.generate(original, sw.DimOrder.HeightWidthChannel, 1000, 0.0)

    # print(len(windows))
    # print(type(windows))

    for index, single_window in enumerate(windows):
        print(single_window)
        x = single_window.x
        y = single_window.y
        width = single_window.w
        height = single_window.h
        cropped_image = original[y:y + height, x:x + width]
        cropped_mask = mask[y:y + height, x:x + width]

        cv2.imwrite(DIRECTORY_CROPPED_IMAGE + 'img_' + str_prefix + str(index) + '.tif', cropped_image)
        cv2.imwrite(DIRECTORY_CROPPED_MASK + 'img_' + str_prefix + str(index) + '.tif', cropped_mask)


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
            #break
    # Calculate number of generated images with masks
    path, dirs, files = next(os.walk(DIRECTORY_CROPPED_MASK))
    file_count = len(files)
    print(f'\n{file_count} images are generated\n')


if __name__ == '__main__':
    main()
