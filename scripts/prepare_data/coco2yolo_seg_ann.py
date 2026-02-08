import json
import os
import pandas as pd
from tqdm import tqdm

def convert_coco_to_yolo_segmentation(json_path, csv_path, output_dir, target_class_id=0):
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Loading annotations from: {json_path}")
    print(f"Loading image list from: {csv_path}")

    with open(json_path, 'r') as f:
        coco_data = json.load(f)

    df = pd.read_csv(csv_path)
    all_filenames = set(df['image_id'].tolist())

    img_id_map = {}
    filename_to_img_id = {}

    for img in coco_data['images']:
        img_id_map[img['id']] = img
        filename_to_img_id[img['file_name']] = img['id']
    
    ann_map = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in ann_map:
            ann_map[img_id] = []
        ann_map[img_id].append(ann)
    
    files_created = 0

    for filename in tqdm(all_filenames, desc="Generating Labels"):

        txt_filename = os.path.splitext(filename)[0] + ".txt"
        output_path = os.path.join(output_dir, txt_filename)

        if filename in filename_to_img_id:
            img_id = filename_to_img_id[filename]

            img_w = img_id_map[img_id]['width']
            img_h = img_id_map[img_id]['height']

            annotations = ann_map.get(img_id, [])

            yolo_lines = []

            for ann in annotations:
                for segment in ann['segmentation']:
                    normalized_coords = []

                    for i in range(0, len(segment), 2):
                        x = segment[i]
                        y = segment[i+1]

                        nx = x / img_w
                        ny = y / img_h

                        nx = max(0.0, min(1.0, nx))
                        ny = max(0.0, min(1.0, ny))

                        normalized_coords.extend([nx, ny])

                    coords_str = " ".join([f"{val:.6f}" for val in normalized_coords])
                    line = f"{target_class_id} {coords_str}"
                    yolo_lines.append(line)

            with open(output_path, 'w') as f:
                f.write("\n".join(yolo_lines))

        else:
            with open(output_path, 'w') as f:
                pass
        
        files_created += 1

    print(f"Conversion completed. Generated {files_created} label files in '{output_dir}'")

if __name__ == "__main__":

    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_script_dir))

    JSON_PATH = os.path.join(project_root, 'FracAtlas', 'Annotations', 'COCO JSON', 'COCO_fracture_masks.json')
    CSV_PATH = os.path.join(project_root, 'FracAtlas', 'dataset.csv')
    OUTPUT_DIR = os.path.join(project_root, 'FracAtlas', 'labels_yolo_seg')
    convert_coco_to_yolo_segmentation(JSON_PATH, CSV_PATH, OUTPUT_DIR)
