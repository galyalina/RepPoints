from __future__ import division
from __future__ import print_function


class TouluseDataSet(path_to_images):

    def __init__(self):
        self.info = {"year": 2020,
                     "version": "1.0",
                     "description": "SemCity Toulouse: A benchmark for building instance segmentation in satellite images",
                     "contributor": "Roscher, Ribana and Volpi, Michele and Mallet, Clément and Drees, Lukas and Wegner, Jan",
                     "url": "http://rs.ipb.uni-bonn.de/data/semcity-toulouse/",
                     "date_created": "2020"
                     }
        self.licenses = [{"id": 1,
                          "name": "Creative Commons Attribution-NonCommercial-ShareAlike 3.0 License",
                          "url": "https://creativecommons.org/licenses/by-nc-sa/3.0/"
                          }]
        self.type = "instances"
        # self.categories = [{"id": seqId + 1, "name": seq["name"], "supercategory": seq["name"]}
        #                    for seqId, seq in enumerate(self.seqs)]

        self.image = {
            "license": 1,
            "file_name": image_name,
            "coco_url": "http://images.cocodataset.org/train2017/000000391895.jpg",
            "height": 460,
            "width": 460,
            "id": image_id
        }


if __name__ == "__main__":
