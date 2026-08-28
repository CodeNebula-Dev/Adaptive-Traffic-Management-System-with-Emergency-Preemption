"""
UA-DETRAC Traffic Surveillance Dataset — Preprocessing & Ingestion Engine.

Processes the UA-DETRAC benchmark (over 1.2M real-world traffic surveillance frames
from overhead intersection CCTV poles) into standardized YOLO 5-class format:

Class Mapping:
    0: car
    1: motorcycle
    2: bus
    3: truck (includes vans and medium/heavy commercial vehicles)
    4: unknown_vehicle (includes 3-wheelers, auto-rickshaws, utility carts, others)

Usage:
    # From local directory containing UA-DETRAC XMLs and images:
    python data/ua_detrac/download_uadetrac.py --data-dir data/ua_detrac

    # Kaggle dataset linking:
    python data/ua_detrac/download_uadetrac.py --kaggle-dir /kaggle/input/ua-detrac-dataset
"""

import os
import sys
import argparse
import glob
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict
import random

# UA-DETRAC vehicle type string mapping to ATMS-Net 5 classes
DETRAC_CATEGORY_MAP = {
    'car': 0,
    'sedan': 0,
    'taxi': 0,
    'suv': 0,
    'motorcycle': 1,
    'motorbike': 1,
    'bicycle': 1,
    'bus': 2,
    'coach': 2,
    'truck': 3,
    'van': 3,
    'pickup': 3,
    'others': 4,
    'other': 4,
    'rickshaw': 4,
    'auto': 4,
    'three_wheeler': 4,
    'unknown_vehicle': 4,
}

CLASS_NAMES = ['car', 'motorcycle', 'bus', 'truck', 'unknown_vehicle']


def parse_detrac_xml(xml_file):
    """
    Parse UA-DETRAC sequence XML annotation file.
    
    Returns:
        frame_annotations: Dict mapping frame_num (int) -> list of [class_id, cx, cy, w, h]
        frame_ignores: Dict mapping frame_num (int) -> list of ignore regions
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    frame_annotations = defaultdict(list)

    # In UA-DETRAC XML:
    # <sequence name="MVI_20011">
    #   <frame num="1">
    #     <target_list>
    #       <target id="1">
    #         <box left="100.0" top="200.0" width="50.0" height="40.0"/>
    #         <attribute vehicle_type="car" speed="0.0"/>
    #       </target>
    #     </target_list>
    #   </frame>
    # </sequence>

    for frame in root.findall('.//frame'):
        frame_num = int(frame.attrib.get('num', 0))
        target_list = frame.find('target_list')
        if target_list is None:
            continue

        for target in target_list.findall('target'):
            box_elem = target.find('box')
            attr_elem = target.find('attribute')
            
            if box_elem is None:
                continue

            left = float(box_elem.attrib.get('left', 0.0))
            top = float(box_elem.attrib.get('top', 0.0))
            w = float(box_elem.attrib.get('width', 0.0))
            h = float(box_elem.attrib.get('height', 0.0))

            v_type = 'car'
            if attr_elem is not None:
                v_type = attr_elem.attrib.get('vehicle_type', 'car').lower()

            class_id = DETRAC_CATEGORY_MAP.get(v_type, 4)

            # Ignore non-positive dimensions
            if w <= 2.0 or h <= 2.0:
                continue

            frame_annotations[frame_num].append({
                'class_id': class_id,
                'left': left,
                'top': top,
                'width': w,
                'height': h,
            })

    return frame_annotations


def process_uadetrac_directory(data_dir, output_dir=None, val_ratio=0.15):
    """
    Scan UA-DETRAC sequence folders and convert to normalized YOLO format.
    """
    if output_dir is None:
        output_dir = data_dir

    labels_dir = os.path.join(output_dir, 'labels')
    os.makedirs(labels_dir, exist_ok=True)

    xml_files = glob.glob(os.path.join(data_dir, '**', '*.xml'), recursive=True)
    if not xml_files:
        # Check for pre-existing YOLO labels
        txt_labels = glob.glob(os.path.join(data_dir, '**', '*.txt'), recursive=True)
        img_files = glob.glob(os.path.join(data_dir, '**', '*.jpg'), recursive=True) + \
                    glob.glob(os.path.join(data_dir, '**', '*.png'), recursive=True)
        
        print(f"Found {len(img_files)} images and {len(txt_labels)} text files in {data_dir}.")
        return create_train_val_splits(img_files, output_dir, val_ratio)

    print(f"Found {len(xml_files)} UA-DETRAC XML annotation sequences.")
    all_processed_images = []
    class_stats = defaultdict(int)

    for xml_path in xml_files:
        seq_name = Path(xml_path).stem.replace('_v3', '')
        frame_dict = parse_detrac_xml(xml_path)

        # Look for sequence image directory
        img_seq_dirs = [
            os.path.join(data_dir, 'Insight-MVT_Annotation_Train', seq_name),
            os.path.join(data_dir, 'Insight-MVT_Annotation_Test', seq_name),
            os.path.join(data_dir, 'images', seq_name),
            os.path.join(data_dir, seq_name),
        ]

        active_img_dir = None
        for p in img_seq_dirs:
            if os.path.exists(p):
                active_img_dir = p
                break

        if active_img_dir is None:
            continue

        # Process frames
        img_files = sorted(glob.glob(os.path.join(active_img_dir, '*.jpg')) + \
                           glob.glob(os.path.join(active_img_dir, '*.png')))

        # Resolution standard for UA-DETRAC is 960x540
        img_w, img_h = 960.0, 540.0

        for img_p in img_files:
            # Extract frame number from filename e.g. img00045.jpg -> 45
            stem = Path(img_p).stem
            digits = ''.join(c for c in stem if c.isdigit())
            if not digits:
                continue
            f_num = int(digits)

            targets = frame_dict.get(f_num, [])
            if not targets:
                continue

            # Write YOLO format label
            label_name = f"{seq_name}_{Path(img_p).name}".replace('.jpg', '.txt').replace('.png', '.txt')
            label_p = os.path.join(labels_dir, label_name)

            yolo_lines = []
            for t in targets:
                cid = t['class_id']
                cx = (t['left'] + t['width'] / 2.0) / img_w
                cy = (t['top'] + t['height'] / 2.0) / img_h
                w = t['width'] / img_w
                h = t['height'] / img_h

                # Clamp to [0, 1]
                cx = min(max(cx, 0.0), 1.0)
                cy = min(max(cy, 0.0), 1.0)
                w = min(max(w, 0.0), 1.0)
                h = min(max(h, 0.0), 1.0)

                yolo_lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                class_stats[cid] += 1

            with open(label_p, 'w') as f:
                f.write('\n'.join(yolo_lines) + '\n')

            all_processed_images.append(img_p)

    print("\n" + "=" * 60)
    print("UA-DETRAC Annotation Processing Summary")
    print("=" * 60)
    print(f"  Total Processed Images: {len(all_processed_images):,}")
    for cid, count in class_stats.items():
        name = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else f'class_{cid}'
        print(f"    ↳ {name:<16}: {count:,} annotations")
    print("=" * 60)

    return create_train_val_splits(all_processed_images, output_dir, val_ratio)


def create_train_val_splits(img_list, output_dir, val_ratio=0.15):
    """Save train.txt and val.txt splits."""
    random.seed(42)
    random.shuffle(img_list)

    n_val = int(len(img_list) * val_ratio)
    val_imgs = img_list[:n_val]
    train_imgs = img_list[n_val:]

    train_txt = os.path.join(output_dir, 'train.txt')
    val_txt = os.path.join(output_dir, 'val.txt')

    with open(train_txt, 'w') as f:
        f.write('\n'.join(train_imgs) + '\n')

    with open(val_txt, 'w') as f:
        f.write('\n'.join(val_imgs) + '\n')

    print(f"\n✓ Generated train split ({len(train_imgs)} images): {train_txt}")
    print(f"✓ Generated val split   ({len(val_imgs)} images):   {val_txt}")
    return train_txt, val_txt


def main():
    parser = argparse.ArgumentParser(description="UA-DETRAC Traffic Surveillance Ingestion")
    parser.add_argument('--data-dir', type=str, default='data/ua_detrac', help="Path to UA-DETRAC dataset root")
    parser.add_argument('--output-dir', type=str, default='data/ua_detrac', help="Path to save YOLO labels & splits")
    parser.add_argument('--val-ratio', type=float, default=0.15, help="Validation fraction (default: 0.15)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    process_uadetrac_directory(args.data_dir, args.output_dir, args.val_ratio)


if __name__ == '__main__':
    main()
