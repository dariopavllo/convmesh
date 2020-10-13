from .pseudo_dataset import BasePseudoDataset

import numpy as np
import torch
import os

class CubPseudoDataset(BasePseudoDataset):
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        
        self.n_classes = (200,)
        args.n_classes = self.n_classes

        # Load CUB labels
        cub_path = 'datasets/cub/CUB_200_2011'

        with open(os.path.join(cub_path, 'images.txt'), 'r') as f:
            images = f.readlines()
            images = [x.split(' ') for x in images]
            ids = {k: v.strip() for k, v in images}

        with open(os.path.join(cub_path, 'image_class_labels.txt'), 'r') as f:
            classes = f.readlines()
            classes = [x.split(' ') for x in classes]
            classes = {k: int(v.strip())-1 for k, v in classes}

        self.filename_to_class = {}
        for k, c in classes.items():
            fname = ids[k]
            self.filename_to_class[fname] = c

        self.classes = [np.array([self.filename_to_class[x]]) for x in self.data['path']]

        num_images = len(self.data['path'])
        if args.conditional_text:
            from utils.text_functions import TextDataProcessorCUB
            
            cub_text_path = cub_path
            self.text_processor = TextDataProcessorCUB(cub_text_path, 'train',
                                                       captions_per_image=10,
                                                       words_num=args.text_max_length)

            self.image_index_to_caption_index = {}
            for ind, el in enumerate(self.data['path']):
                self.image_index_to_caption_index[ind] = self.text_processor.filenames_to_index[el]

            # Randomly select a sentence for evaluation
            np.random.seed(1234)
            sent_ix = np.random.randint(0, 10)
            self.index_captions = [self.text_processor.get_caption(self.image_index_to_caption_index[idx_gt] *\
                                   self.text_processor.embeddings_num+sent_ix, words_num=25) for idx_gt in range(num_images)]
            
        print('Loaded CUB dataset with {} images and {} classes'.format(num_images, self.n_classes))
    
    def name(self):
        return 'cub'
    
    def suggest_truncation_sigma(self):
        args = self.args
        if args.conditional_class:
            return 0.25
        elif args.conditional_text:
            return 0.5
        else: # Unconditional
            return 1.0
        
    def suggest_num_discriminators(self):
        if self.args.texture_resolution >= 512:
            return 3
        else:
            return 2
    
    def suggest_mesh_template(self):
        return 'mesh_templates/uvsphere_16rings.obj'
    
    def get_random_caption(self, idx):
        # Randomly select a sentence belonging to image idx
        sent_ix = torch.randint(0, self.text_processor.embeddings_num, size=(1,)).item()
        new_sent_ix = self.image_index_to_caption_index[idx] * self.text_processor.embeddings_num + sent_ix
        return self.text_processor.get_caption(new_sent_ix) # Tuple (padded tokens, lengths)
    
    def __getitem__(self, idx):
        gt_dict = super().__getitem__(idx)
        
        if self.args.conditional_text:
            gt_dict['caption'] = self.get_random_caption(idx)
            
        return gt_dict