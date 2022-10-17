from kmeans import AnchorKmeans
from datasets import AnnotParser
import argparse


def main(args):
    file_type = args["type"]
    k = args["k_clusters"]
    annot_dir = args["dir_path"]
    parser = AnnotParser(file_type)

    print("[INFO] Load datas from {}".format(annot_dir))
    boxes = parser.parse(annot_dir)

    print("[INFO] Initialize model")
    model = AnchorKmeans(k)

    print("[INFO] Training...")
    model.fit(boxes)

    anchors = model.anchors_
    print("[INFO] The results anchors:\n{}".format(anchors))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir_path",
                    default='./Train/Annotations',
                    required=True,
                    help="directory path of annotation files")#"-d","--dir_path",
    ap.add_argument("--type",
                    choices=['xml', 'json', 'csv'],
                    default='xml',
                    help="type of annotation file")#"-t",
    ap.add_argument("--k_clusters",
                    type=int,
                    default=3,
                    help="the number of clusters")#"-k",
    args = vars(ap.parse_args())
    main(args)
