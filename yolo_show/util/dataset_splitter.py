import os
import shutil
import json
import random


def split_coco_dataset(dataset_dir, train_ratio, val_ratio, test_ratio):
    image_dir = os.path.join(dataset_dir, 'images')
    annotation_file = os.path.join(dataset_dir, 'annotations', 'instances_train2017.json')

    with open(annotation_file, 'r') as f:
        coco_data = json.load(f)

    images = coco_data['images']
    random.shuffle(images)

    num_images = len(images)
    train_end = int(train_ratio * num_images)
    val_end = train_end + int(val_ratio * num_images)

    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    output_dirs = {
        'train': os.path.join(dataset_dir, 'train'),
        'val': os.path.join(dataset_dir, 'val'),
        'test': os.path.join(dataset_dir, 'test')
    }

    for key in output_dirs:
        os.makedirs(output_dirs[key], exist_ok=True)
        os.makedirs(os.path.join(output_dirs[key], 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dirs[key], 'annotations'), exist_ok=True)

    split_data = {
        'train': train_images,
        'val': val_images,
        'test': test_images
    }

    for key in split_data:
        for image_info in split_data[key]:
            image_id = image_info['id']
            image_filename = image_info['file_name']

            src_image_path = os.path.join(image_dir, image_filename)
            dst_image_path = os.path.join(output_dirs[key], 'images', image_filename)
            shutil.copy(src_image_path, dst_image_path)

        new_annotations = {
            'images': split_data[key],
            'annotations': [ann for ann in coco_data['annotations'] if
                            ann['image_id'] in {img['id'] for img in split_data[key]}],
            'categories': coco_data['categories']
        }

        annotation_output_file = os.path.join(output_dirs[key], 'annotations', 'instances_{}.json'.format(key))
        with open(annotation_output_file, 'w') as f:
            json.dump(new_annotations, f)

