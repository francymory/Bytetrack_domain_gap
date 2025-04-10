import os
import os.path as osp
import json
import argparse
import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--motsynth-path', help="Directory path containing the 'annotations' directory with .json files")
    parser.add_argument('--save-path', help='Root file in which the new annoation files will be stored. If not provided, motsynth-root will be used')
    parser.add_argument('--save-dir', default='comb_annotations', help="name of directory within 'save-path'in which MOTS annotation files will be stored")
    parser.add_argument('--subsample', default=10, type=int, help="Frame subsampling rate. If e.g. 10 is selected, then we will select 1 in 10 frames")
    parser.add_argument('--split', default='train', help="Name of split (i.e. set of sequences being merged) being used. A file named '{args.split}.txt needs to exist in the splits dir")
    parser.add_argument('--name', help="Name of the split that file that will be generated. If not provided, the split name will be used")
    
    args = parser.parse_args()

    if args.save_path is None:
        args.save_path = args.motsynth_path
    
    if args.name is None:
        args.name = args.split

    assert args.subsample >0, "Argument '--subsample' needs to be a positive integer. Set it to 1 to use every frame"

    return args
    
def read_split_file(path):
    with open(path, 'r') as file:
        seq_list = file.read().splitlines()

    return seq_list

def main(args):    
    # Determine which sequences to use
    seqs = [seq.zfill(3) for seq in read_split_file(osp.join(args.motsynth_path, 'seqmaps', f'{args.split}.txt'))]
    comb_anns  = {'images': [], 'annotations': [], 'categories': None, 'info': {
            "description": "MOTSynth 10",
            "url": "http://aimagelab.ing.unimore.it/jta",
            "version": "1.0",
            "year": 2021,
            "contributor": "AImageLab",
            "date_created": "2021/01/22"
        }}

    for seq in tqdm.tqdm(seqs):
        ann_path = osp.join(args.motsynth_path, 'annotations',  f'{seq}.json')
        with open(ann_path) as f:
            seq_ann = json.load(f)

        # Subsample images and annotations if needed
        if args.subsample >1:
            seq_ann['images'] = [{**img, **seq_ann['info']} for img in seq_ann['images'] if ((img['frame_n'] -1 )% args.subsample) == 0] # -1 bc in the paper this was 0-based 
            img_ids = [img['id'] for img in seq_ann['images']]
            seq_ann['annotations'] = [ann for ann in seq_ann['annotations'] if ann['image_id'] in img_ids]

#MODIFICA:

        # Modifica delle annotazioni
        for ann in seq_ann['annotations']:
            # Rimuovi le chiavi inutili
            ann.pop("num_keypoints", None)
            ann.pop('segmentation', None)
            ann.pop("model_id", None)
            ann.pop("attributes", None)
            ann.pop("is_blurred", None)
            

            # Calcola visibility: 1 se almeno un keypoint ha valore 2, altrimenti 0
            visibility = 0
            keypoints = ann.get('keypoints', [])
            for i in range(0, len(keypoints), 3):  # I keypoints sono in formato (x, y, visibility)
                if keypoints[i+2] == 2:  # La visibilità è nel terzo elemento (indice 2)
                    visibility = 1
                    break  # Una volta trovato un keypoint con visibilità 2, non è necessario continuare

            # Aggiungi il campo visibility
            ann['visibility'] = visibility
            # Rimuovi i keypoints originali
            ann.pop('keypoints', None)


            # Calcola la distanza media dalla telecamera dei keypoints_3d visibili
            keypoints_3d = ann.get('keypoints_3d', [])
            distances = []
            for i in range(0, len(keypoints_3d), 4):  # formato: x, y, z, visibilità (se è 2, il punto è visibile)
                x, y, z, v = keypoints_3d[i:i+4]
                if v > 1:
                    distance = (x**2 + y**2 + z**2) ** 0.5
                    distances.append(distance)

            if distances:
                avg_distance = sum(distances) / len(distances)
            else:
                avg_distance = float('inf')  # Se non ci sono keypoint visibili, metti infinito

            # Aggiungi il campo 'distance' e rimuovi keypoints_3d
            ann['distance'] = avg_distance
            ann.pop('keypoints_3d', None)


        #Modifica di images
        for img in seq_ann['images']:
            img.pop("cam_world_pos", None)
            img.pop("cam_world_rot", None)
            img.pop("ignore_mask", None)
            img.pop("description", None)
            img.pop("version", None)
            img.pop("img_width", None)
            img.pop("img_height", None)
            img.pop("is_night", None)
            img.pop("is_moving", None)
            img.pop("weather", None)
            img.pop("cam_fov", None)
            img.pop("fps", None)
            img.pop("sequence_length", None)
            img.pop("time", None)
            img.pop("fx", None)
            img.pop("fy", None)
            img.pop("cx", None)
            img.pop("cy", None)
                
#Fine modifica

        comb_anns['images'].extend(seq_ann['images'])
        comb_anns['annotations'].extend(seq_ann['annotations'])

    
    if len(seqs) > 0:
        comb_anns['categories'] = seq_ann['categories']
        comb_anns['licenses'] = seq_ann['categories']

    # Sanity check:
    img_ids = [img['id'] for img in comb_anns['images']]
    ann_ids = [ann['id'] for ann in comb_anns['annotations']]
    assert len(img_ids) == len(set(img_ids))
    assert len(ann_ids) == len(set(ann_ids))

    # Save the new annotations file
    comb_anns_dir = osp.join(args.save_path, args.save_dir)
    os.makedirs(comb_anns_dir, exist_ok=True)
    comb_anns_path = osp.join(comb_anns_dir, f"{args.name}.json")
    with open(comb_anns_path, 'w') as json_file:
        json.dump(comb_anns, json_file)
    

if __name__ == '__main__':
    args = parse_args()
    main(args)


'''import os
import os.path as osp
import json
import argparse
import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--motsynth-path', help="Directory path containing the 'annotations' directory with .json files")
    parser.add_argument('--save-path', help='Root file in which the new annoation files will be stored. If not provided, motsynth-root will be used')
    parser.add_argument('--save-dir', default='comb_annotations', help="name of directory within 'save-path'in which MOTS annotation files will be stored")
    parser.add_argument('--subsample', default=10, type=int, help="Frame subsampling rate. If e.g. 10 is selected, then we will select 1 in 10 frames")
    parser.add_argument('--split', default='train', help="Name of split (i.e. set of sequences being merged) being used. A file named '{args.split}.txt needs to exist in the splits dir")
    parser.add_argument('--name', help="Name of the split that file that will be generated. If not provided, the split name will be used")
    
    args = parser.parse_args()

    if args.save_path is None:
        args.save_path = args.motsynth_path
    
    if args.name is None:
        args.name = args.split

    assert args.subsample >0, "Argument '--subsample' needs to be a positive integer. Set it to 1 to use every frame"

    return args
    
def read_split_file(path):
    with open(path, 'r') as file:
        seq_list = file.read().splitlines()

    return seq_list

def main(args):    
    # Determine which sequences to use
    seqs = [seq.zfill(3) for seq in read_split_file(osp.join(args.motsynth_path, 'seqmaps', f'{args.split}.txt'))]
    comb_anns  = {'images': [], 'annotations': [], 'categories': None, 'info': {
            "description": "MOTSynth 10",
            "url": "http://aimagelab.ing.unimore.it/jta",
            "version": "1.0",
            "year": 2021,
            "contributor": "AImageLab",
            "date_created": "2021/01/22"
        }}

    for seq in tqdm.tqdm(seqs):
        ann_path = osp.join(args.motsynth_path, 'annotations',  f'{seq}.json')
        with open(ann_path) as f:
            seq_ann = json.load(f)

        # Subsample images and annotations if needed
        if args.subsample >1:
            seq_ann['images'] = [{**img, **seq_ann['info']} for img in seq_ann['images'] if ((img['frame_n'] -1 )% args.subsample) == 0] # -1 bc in the paper this was 0-based 
            img_ids = [img['id'] for img in seq_ann['images']]
            seq_ann['annotations'] = [ann for ann in seq_ann['annotations'] if ann['image_id'] in img_ids]

#MODIFICA:

        # Modifica delle annotazioni
        for ann in seq_ann['annotations']:
            # Rimuovi la chiave 'segmentation' e 'keypoints_3d'
            ann.pop('segmentation', None)
            ann.pop('keypoints_3d', None)
            
            # Calcola visibility: 1 se almeno un keypoint ha valore 2, altrimenti 0
            visibility = 0
            keypoints = ann.get('keypoints', [])
            for i in range(0, len(keypoints), 3):  # I keypoints sono in formato (x, y, visibility)
                if keypoints[i+2] == 2:  # La visibilità è nel terzo elemento (indice 2)
                    visibility = 1
                    break  # Una volta trovato un keypoint con visibilità 2, non è necessario continuare

            # Aggiungi il campo visibility
            ann['visibility'] = visibility
            # Rimuovi i keypoints originali
            ann.pop('keypoints', None)
                
#Fine modifica

        comb_anns['images'].extend(seq_ann['images'])
        comb_anns['annotations'].extend(seq_ann['annotations'])
        #comb_anns['info'][seq] = seq_ann['info']
    
    if len(seqs) > 0:
        comb_anns['categories'] = seq_ann['categories']
        comb_anns['licenses'] = seq_ann['categories']

    # Sanity check:
    img_ids = [img['id'] for img in comb_anns['images']]
    ann_ids = [ann['id'] for ann in comb_anns['annotations']]
    assert len(img_ids) == len(set(img_ids))
    assert len(ann_ids) == len(set(ann_ids))

    # Save the new annotations file
    comb_anns_dir = osp.join(args.save_path, args.save_dir)
    os.makedirs(comb_anns_dir, exist_ok=True)
    comb_anns_path = osp.join(comb_anns_dir, f"{args.name}.json")
    with open(comb_anns_path, 'w') as json_file:
        json.dump(comb_anns, json_file)
    

if __name__ == '__main__':
    args = parse_args()
    main(args)
    '''