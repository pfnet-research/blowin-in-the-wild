import cv2
import os
from glob import glob
from argparse import ArgumentParser
from tqdm import tqdm


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--factor", type=int, default=4)
    args = parser.parse_args()

    images_dir = os.path.join(args.input_dir, "images")
    img_list = glob(os.path.join(images_dir, "*.JPG"))
    for img in tqdm(img_list):
        fname = os.path.basename(img)
        img_org = cv2.imread(img)
        img_resized = cv2.resize(img_org, None, fx=1/args.factor,fy=1/args.factor)
        
        save_dir = os.path.join(args.output_dir, f"images_{args.factor}")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        cv2.imwrite(os.path.join(save_dir, fname), img_resized)
