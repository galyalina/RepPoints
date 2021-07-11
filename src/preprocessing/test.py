import fiftyone as fo
import glob
import json

DIRECTORY_IMAGE = "../../data_test/train/"
ANNOTATIONS = "../../data_test/annotations/coco.json"

if __name__ == '__main__':
    images_patt = DIRECTORY_IMAGE + "/*"

    # Ex: your custom label format
    with open(ANNOTATIONS) as coco:
        annotations = json.load(coco)['annotations']

    # Create dataset
    dataset = fo.Dataset(name="test-dataset-3")

    # Persist the dataset on disk in order to
    # be able to load it in one line in the future
    dataset.persistent = True

    # Add your samples to the dataset
    for filepath in glob.glob(images_patt):
        sample = fo.Sample(filepath=filepath)

        # Convert detections to FiftyOne format
        detections = []
        for obj in annotations:
            label = str(obj["category_id"])

            # Bounding box coordinates should be relative values
            # in [0, 1] in the following format:
            # [top-left-x, top-left-y, width, height]
            bounding_box = obj["bbox"]

            detections.append(
                fo.Detection(label=label, bounding_box=bounding_box)
            )

        # Store detections in a field name of your choice
        sample["ground_truth"] = fo.Detections(detections=detections)

        dataset.add_sample(sample)

    session = fo.launch_app(dataset)