import torch
import torch.utils.data as data
import os
from tqdm import tqdm
import numpy as np
import copy
from .helpers.psutil import FreeMemLinux
import clip
import random
import pickle

changed_relationships_dict = {
        'left': 'right',
        'right': 'left',
        'front': 'behind',
        'behind': 'front',
        'bigger than': 'smaller than',
        'smaller than': 'bigger than',
        'taller than': 'shorter than',
        'shorter than': 'taller than',
        'close by': 'close by',
        'same style as': 'same style as',
        'same super category as': 'same super category as',
        'same material as': 'same material as',
        "symmetrical to": "symmetrical to",
        "standing on":"standing on",
        "above":"above"
    }

def load_ckpt(ckpt):
    map_fn = lambda storage, loc: storage
    if type(ckpt) == str:
        state_dict = torch.load(ckpt, map_location=map_fn)
    else:
        state_dict = ckpt
    return state_dict

class nuScenesLayout(data.Dataset):
    def __init__(self, root, split='train', shuffle_objs=False,
                 use_scene_rels=False, data_len=None, with_changes=True, scale_func='diag', eval=False,
                 eval_type='addition', with_CLIP=False, bin_angle=False,
                 seed=True, recompute_feats=False, recompute_clip=False, dataset='nuscenes'):
        self.dataset = dataset
        
        self.seed = seed
        self.with_CLIP = with_CLIP
        self.cond_model = None
        self.recompute_feats = recompute_feats
        self.recompute_clip = recompute_clip

        if eval and seed:
            np.random.seed(47)
            torch.manual_seed(47)
            random.seed(47)

        self.scale_func = scale_func
        self.with_changes = with_changes
        self.root = root
        self.catfile = os.path.join(self.root,'classes_{}.txt'.format(self.dataset))
        self.cat = {}
        self.scans = []
        self.obj_paths = []
        self.data_len = data_len
        self.use_scene_rels = use_scene_rels
        self.bin_angle = bin_angle
        self.box_range = [-50,-50,-3,50,50,1]
        self.fm = FreeMemLinux('GB')
        self.vocab = {}
        with open(os.path.join(self.root, 'classes_{}.txt'.format(self.dataset)), "r") as f:
            self.vocab['object_idx_to_name'] = f.readlines()
        with open(os.path.join(self.root, 'relationships.txt'), "r") as f:
            self.vocab['pred_idx_to_name'] = ['in\n']
            self.vocab['pred_idx_to_name']+=f.readlines()

        # list of relationship categories
        self.relationships = self.read_relationships(os.path.join(self.root, 'relationships.txt'))
        self.relationships_dict = dict(zip(self.relationships,range(len(self.relationships))))
        self.relationships_dict_r = dict(zip(self.relationships_dict.values(), self.relationships_dict.keys()))

        if split == 'train':
            self.training = True
            self.rel_box_json_file = os.path.join(self.root, 'nuscenes_infos_train.pkl')
        else: # test set
            self.training = False
            self.rel_box_json_file = os.path.join(self.root, 'nuscenes_infos_val.pkl')

        self.relationship_json, self.objs_json, self.tight_boxes_json = \
                self.read_relationship_json(self.rel_box_json_file)

        
        self.eval = eval
        self.shuffle_objs = shuffle_objs

        with open(self.catfile, 'r') as f:
            for line in f:
                category = line.rstrip()
                self.cat[category] = category

        self.classes = dict(zip(self.cat, range(len(self.cat))))
        self.classes_r = dict(zip(self.classes.values(), self.classes.keys()))

        self.eval_type = eval_type

        # check if all clip features exist. If not they get generated here (once)
        if self.with_CLIP:
            self.cond_model, preprocess = clip.load("ViT-B/32", device='cuda')
            self.cond_model_cpu, preprocess_cpu = clip.load("ViT-B/32", device='cpu')
            print('loading CLIP')
            print('Checking for missing clip feats. This can be slow the first time.')
            for index in tqdm(range(len(self))):
                self.__getitem__(index)
            self.recompute_clip = False

    def read_relationship_json(self, rel_box_json_file):
        """ Reads from json files the relationship labels, objects and bounding boxes

        :param json_file: file that stores the objects and relationships
        :param box_json_file: file that stores the oriented 3D bounding box parameters
        :return: three dicts, relationships, objects and boxes
        """
        rel = {}
        objs = {}
        tight_boxes = {}
        lidar_path = {}
        with open(rel_box_json_file, 'rb') as f:
            data_infos = pickle.load(f)
        for i in range(len(data_infos)):
            frame_id = str(i).zfill(7)
            self.scans.append(frame_id)
            info = data_infos[i]
            rel[frame_id] = info['scene_graph']['keep_box_relationships']
            objs[frame_id] = info['scene_graph']['keep_box_names']
            tight_boxes[frame_id] = info['scene_graph']['keep_box']
        return rel, objs, tight_boxes

    def read_relationships(self, read_file):
        """load list of relationship labels

        :param read_file: path of relationship list txt file
        """
        relationships = []
        with open(read_file, 'r') as f:
            for line in f:
                relationship = line.rstrip().lower()
                relationships.append(relationship)
        return relationships

    def get_key(self, dict, value):
        for k, v in dict.items():
            if v == value:
                return k
        return None

    def scale_box(self, boxes):
        boxes = np.array(boxes)
        new_boxes = np.zeros([boxes.shape[0] + 1, 7])
        x_min,y_min,z_min,x_max,y_max,z_max = self.box_range
        boxes[:,0] = (boxes[:,0] - x_min) / (x_max - x_min)
        boxes[:,1] = (boxes[:,1] - y_min) / (y_max - y_min)
        boxes[:,2] = (boxes[:,2] - z_min) / (z_max - z_min)
        boxes[:,3:6] = np.log(boxes[:,3:6])
        new_boxes[1:,:7] = boxes[:,:7]
        new_boxes[0,:] = -1
        return new_boxes

    def re_scale_box(self, boxes):
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.detach().cpu().numpy()
        x_min,y_min,z_min,x_max,y_max,z_max = self.box_range
        boxes[1:,0] = boxes[1:,0] * (x_max - x_min) + x_min
        boxes[1:,1] = boxes[1:,1] * (y_max - y_min) + y_min
        boxes[1:,2] = boxes[1:,2] * (z_max - z_min) + z_min
        boxes[1:,3:6] = np.exp(boxes[1:,3:6])
        boxes[0,:] = 0
        return boxes

    def __getitem__(self, index):
        scan_id = self.scans[index]
        # If true, expected paths to saved clip features will be set here
        if self.with_CLIP:
            if_training = 'train' if self.training else 'val'
            self.clip_feats_path = os.path.join(self.root, if_training, 'CLIP', scan_id,
                                                'CLIP_{}.pkl'.format(scan_id))
            if not os.path.exists(os.path.join(self.root, if_training, 'CLIP', scan_id)):
                os.makedirs(os.path.join(self.root, if_training, 'CLIP', scan_id))
            if self.recompute_clip:
                self.clip_feats_path += 'tmp'

        boxes = self.tight_boxes_json[scan_id]
        boxes = self.scale_box(boxes)

        triples = []
        words = []
        rel_json = self.relationship_json[scan_id]
        obj_json = ['ego'] + list(self.objs_json[scan_id])

        objs_count = {}
        unique_obj_json = []

        for obj in obj_json:
            if obj_json.count(obj) > 1:
                count = objs_count.get(obj, 0) + 1
                objs_count[obj] = count
                new_obj = f"{obj}{count}"
                unique_obj_json.append(new_obj)
            else:
                unique_obj_json.append(obj)

        for r in rel_json:
            triples.append(r)
            words.append(unique_obj_json[r[0]] + ' ' + self.relationships[r[1]] + ' ' + unique_obj_json[r[2]]) # TODO check

        if self.with_CLIP:
            # If precomputed features exist, we simply load them
            if os.path.exists(self.clip_feats_path):
                clip_feats_dic = pickle.load(open(self.clip_feats_path, 'rb'))
                clip_feats_ins = clip_feats_dic['instance_feats']
                clip_feats_rel = clip_feats_dic['rel_feats']
                if not isinstance(clip_feats_ins, list):
                    clip_feats_ins = list(clip_feats_ins)

        output = {}
        # if features are requested but the files don't exist, we run all loaded cats and triples through clip
        # to compute them and then save them for future usage
        if self.with_CLIP and (not os.path.exists(self.clip_feats_path) or clip_feats_ins is None) and self.cond_model is not None:
            feats_rel = {}
            with torch.no_grad():

                text_obj = clip.tokenize(obj_json).to('cuda')
                feats_ins = self.cond_model.encode_text(text_obj).detach().cpu().numpy()
                text_rel = clip.tokenize(words).to('cuda')
                rel = self.cond_model.encode_text(text_rel).detach().cpu().numpy()
                for i in range(len(words)):
                    feats_rel[words[i]] = rel[i]

            clip_feats_in = {}
            clip_feats_in['instance_feats'] = feats_ins
            clip_feats_in['rel_feats'] = feats_rel
            path = os.path.join(self.clip_feats_path)
            if self.recompute_clip:
                path = path[:-3]

            pickle.dump(clip_feats_in, open(path, 'wb'))
            clip_feats_ins = list(clip_feats_in['instance_feats'])
            clip_feats_rel = clip_feats_in['rel_feats']

        # prepare outputs
        output['encoder'] = {}
        output['encoder']['objs'] = [self.classes[obj] for obj in obj_json]
        output['encoder']['triples'] = triples
        output['encoder']['boxes'] = list(boxes)
        output['encoder']['words'] = words

        if self.with_CLIP:
            output['encoder']['text_feats'] = clip_feats_ins
            clip_feats_rel_new = []
            if clip_feats_rel != None:
                for word in words:
                    clip_feats_rel_new.append(clip_feats_rel[word])
                output['encoder']['rel_feats'] = clip_feats_rel_new

        output['manipulate'] = {}
        if not self.with_changes:
            output['manipulate']['type'] = 'none'
            output['decoder'] = copy.deepcopy(output['encoder'])
        else:
            if not self.eval:
                if self.with_changes:
                    output['manipulate']['type'] = ['relationship', 'addition', 'none'][
                        # 1]
                        np.random.randint(3)]  # removal is trivial - so only addition and rel change
                else:
                    output['manipulate']['type'] = 'none'
                output['decoder'] = copy.deepcopy(output['encoder'])
                if len(output['encoder']['objs']) <=2:
                    output['manipulate']['type'] = 'none'
                if output['manipulate']['type'] == 'addition':
                    node_id, node_removed, node_clip_removed, triples_removed, triples_clip_removed, words_removed = self.remove_node_and_relationship(output['encoder'])
                    if node_id >= 0:
                        output['manipulate']['added_node_id'] = node_id
                        output['manipulate']['added_node_class'] = node_removed
                        output['manipulate']['added_node_clip'] = node_clip_removed
                        output['manipulate']['added_triples'] = triples_removed
                        output['manipulate']['added_triples_clip'] = triples_clip_removed
                        output['manipulate']['added_words'] = words_removed
                    else:
                        output['manipulate']['type'] = 'none'
                elif output['manipulate']['type'] == 'relationship':
                    rel, original_triple, suc = self.modify_relship(output['encoder']) # why modify encoder side? Because the changed edge doesn't need to make sense. I need to make sure that the one from the decoder side makes sense
                    if suc:
                        output['manipulate']['original_relship'] = (rel, original_triple)
                    else:
                        output['manipulate']['type'] = 'none'
            else:
                output['manipulate']['type'] = self.eval_type
                output['decoder'] = copy.deepcopy(output['encoder'])
                if output['manipulate']['type'] == 'addition':
                    node_id, node_removed, node_clip_removed, triples_removed, triples_clip_removed, words_removed = self.remove_node_and_relationship(output['encoder'])
                    if node_id >= 0:
                        output['manipulate']['added_node_id'] = node_id
                        output['manipulate']['added_node_class'] = node_removed
                        output['manipulate']['added_node_clip'] = node_clip_removed
                        output['manipulate']['added_triples'] = triples_removed
                        output['manipulate']['added_triples_clip'] = triples_clip_removed
                        output['manipulate']['added_words'] = words_removed
                    else:
                        return -1
                elif output['manipulate']['type'] == 'relationship':
                    # this should be modified from the decoder side, because during test we have to evaluate the real setting, and can make the change have the real meaning.
                    rel, original_triple, suc = self.modify_relship(output['decoder'], interpretable=True)
                    if suc:
                        output['manipulate']['original_relship'] = (rel, original_triple)
                    else:
                        return -1

        # torchify
        output['encoder']['objs'] = torch.from_numpy(np.array(output['encoder']['objs'], dtype=np.int64)) # this is changed
        output['encoder']['triples'] = torch.from_numpy(np.array(output['encoder']['triples'], dtype=np.int64))
        output['encoder']['boxes'] = torch.from_numpy(np.array(output['encoder']['boxes'], dtype=np.float32))
        if self.with_CLIP:
            output['encoder']['text_feats'] = torch.from_numpy(np.array(output['encoder']['text_feats'], dtype=np.float32)) # this is changed
            output['encoder']['rel_feats'] = torch.from_numpy(np.array(output['encoder']['rel_feats'], dtype=np.float32))

        # these two should have the same amount.
        output['decoder']['objs'] = torch.from_numpy(np.array(output['decoder']['objs'], dtype=np.int64))

        output['decoder']['triples'] = torch.from_numpy(np.array(output['decoder']['triples'], dtype=np.int64)) # this is changed
        output['decoder']['boxes'] = torch.from_numpy(np.array(output['decoder']['boxes'], dtype=np.float32))
        if self.with_CLIP:
            output['decoder']['text_feats'] = torch.from_numpy(np.array(output['decoder']['text_feats'], dtype=np.float32))
            output['decoder']['rel_feats'] = torch.from_numpy(np.array(output['decoder']['rel_feats'], dtype=np.float32)) # this is changed

        output['scan_id'] = scan_id

        return output

    def remove_node_and_relationship(self, graph):
        """ Automatic random removal of certain nodes at training time to enable training with changes. In that case
        also the connecting relationships of that node are removed

        :param graph: dict containing objects, features, boxes, points and relationship triplets
        :return: index of the removed node
        """

        node_id = -1
        # dont remove layout components, like floor. those are essential
        excluded = [self.classes['ego']]

        trials = 0
        while node_id < 0 or graph['objs'][node_id] in excluded:
            if trials > 100:
                return -1
            trials += 1
            node_id = np.random.randint(len(graph['objs']) - 1)

        node_removed = graph['objs'].pop(node_id)
        node_clip_removed = None
        if self.with_CLIP:
            node_clip_removed = graph['text_feats'].pop(node_id)

        graph['boxes'].pop(node_id)

        to_rm = []
        triples_clip_removed_list = []
        words_removed_list = []
        for i,x in reversed(list(enumerate(graph['triples']))):
            sub, pred, obj = x
            if sub == node_id or obj == node_id:
                to_rm.append(x)
                if self.with_CLIP:
                    triples_clip_removed_list.append(graph['rel_feats'].pop(i))
                    words_removed_list.append(graph['words'].pop(i))

        triples_removed = copy.deepcopy(to_rm)
        while len(to_rm) > 0:
            graph['triples'].remove(to_rm.pop(0))

        for i in range(len(graph['triples'])):
            if graph['triples'][i][0] > node_id:
                graph['triples'][i][0] -= 1

            if graph['triples'][i][2] > node_id:
                graph['triples'][i][2] -= 1

        # node_id: instance_id; node_removed: class_id; triples_removed: relations (sub_id, edge_id, obj_id)
        return node_id, node_removed, node_clip_removed, triples_removed, triples_clip_removed_list, words_removed_list

    def modify_relship(self, graph, interpretable=False):
        """ Change a relationship type in a graph

        :param graph: dict containing objects, features, boxes, points and relationship triplets
        :param interpretable: boolean, if true choose a subset of easy to interpret relations for the changes
        :return: index of changed triplet, a tuple of affected subject & object, and a boolean indicating if change happened
        """

        # rels 26 -> 0
        '''15 same material as' '14 same super category as' '13 same style as' '12 symmetrical to' '11 shorter than' '10 taller than' '9 smaller than'
         '8 bigger than' '7 standing on' '6 above' '5 close by' '4 behind' '3 front' '2 right' '1 left'
         '0: none'''
        # subset of edge labels that are spatially interpretable (evaluatable via geometric contraints)
        interpretable_rels = [0, 1, 2, 3, 5, 6, 7, 8]
        did_change = False
        trials = 0
        excluded = []
        eval_excluded = []

        while not did_change and trials < 1000:
            idx = np.random.randint(len(graph['triples']))
            sub, pred, obj = graph['triples'][idx]
            trials += 1

            if graph['objs'][obj] in excluded or graph['objs'][sub] in excluded:
                continue
            if interpretable:
                if graph['objs'][obj] in eval_excluded or graph['objs'][sub] in eval_excluded: # don't use the floor
                    continue
                if pred not in interpretable_rels:
                    continue
                else:
                    new_pred = self.relationships_dict[changed_relationships_dict[self.relationships_dict_r[pred]]]
            else:
                new_pred = np.random.randint(0, 9)
                if new_pred == pred:
                    continue

            graph['words'][idx] = graph['words'][idx].replace(self.relationships_dict_r[graph['triples'][idx][1]],self.relationships_dict_r[new_pred])
            graph['changed_id'] = idx

            # When interpretable is false, we can make things from the encoder side not existed, so that make sure decoder side is the real data.
            # When interpretable is true, we can make things from the decoder side not existed, so that make sure what we test (encoder side) is the real data.
            graph['triples'][idx][1] = new_pred # this new_pred may even not exist.

            did_change = True

        # idx: idx-th triple; (sub, pred, obj): previous triple
        return idx, (sub, pred, obj), did_change

    def __len__(self):
        if self.data_len is not None:
            return self.data_len
        else:
            return len(self.scans)


    def collate_fn(self, batch):
        """
        Collate function to be used when wrapping a RIODatasetSceneGraph in a
        DataLoader. Returns a dictionary
        """

        out = {}

        out['scene_points'] = []
        out['scan_id'] = []
        out['instance_id'] = []

        out['missing_nodes'] = []
        out['added_nodes'] = []
        out['missing_nodes_decoder'] = []
        out['manipulated_subs'] = []
        out['manipulated_objs'] = []
        out['manipulated_preds'] = []
        out['range_image'] = []
        out['range_mask'] = []
        global_node_id = 0
        global_dec_id = 0
        for i in range(len(batch)):
            if batch[i] == -1:
                return -1
            # notice only works with single batches
            out['scan_id'].append(batch[i]['scan_id'])
            # out['instance_id'].append(batch[i]['instance_id'])

            if batch[i]['manipulate']['type'] == 'addition':
                out['missing_nodes'].append(global_node_id + batch[i]['manipulate']['added_node_id'])
                out['added_nodes'].append(global_dec_id + batch[i]['manipulate']['added_node_id'])
            elif batch[i]['manipulate']['type'] == 'relationship':
                rel, (sub, pred, obj) = batch[i]['manipulate']['original_relship'] # remember that this is already changed in the initial scene graph, which means this triplet is real data.
                # which node is changed in the beginning.
                out['manipulated_subs'].append(global_node_id + sub)
                out['manipulated_objs'].append(global_node_id + obj)
                out['manipulated_preds'].append(pred) # this is the real edge

            global_node_id += len(batch[i]['encoder']['objs'])
            global_dec_id += len(batch[i]['decoder']['objs'])

        for key in ['encoder', 'decoder']:
            all_objs, all_boxes, all_triples = [], [], []
            all_obj_to_scene, all_triple_to_scene = [], []
            all_points = []
            all_sdfs = []
            all_text_feats = []
            all_rel_feats = []

            obj_offset = 0

            for i in range(len(batch)):
                if batch[i] == -1:
                    print('this should not happen')
                    continue
                (objs, triples, boxes) = batch[i][key]['objs'], batch[i][key]['triples'], batch[i][key]['boxes']

                if 'points' in batch[i][key]:
                    all_points.append(batch[i][key]['points'])
                elif 'sdfs' in batch[i][key]:
                    all_sdfs.append(batch[i][key]['sdfs'])
                if 'text_feats' in batch[i][key]:
                    all_text_feats.append(batch[i][key]['text_feats'])
                if 'rel_feats' in batch[i][key]:
                    if 'changed_id' in batch[i][key]:
                        idx = batch[i][key]['changed_id']
                        if self.with_CLIP:
                            text_rel = clip.tokenize(batch[i][key]['words'][idx]).to('cpu')
                            rel = self.cond_model_cpu.encode_text(text_rel).detach().numpy()
                            batch[i][key]['rel_feats'][idx] = torch.from_numpy(np.squeeze(rel)) # this should be a fake relation from the encoder side

                    all_rel_feats.append(batch[i][key]['rel_feats'])

                num_objs, num_triples = objs.size(0), triples.size(0)

                all_objs.append(batch[i][key]['objs'])
                # all_objs_grained.append(batch[i][key]['objs_grained'])
                all_boxes.append(boxes)

                if triples.dim() > 1:
                    triples = triples.clone()
                    triples[:, 0] += obj_offset
                    triples[:, 2] += obj_offset

                    all_triples.append(triples)
                    all_triple_to_scene.append(torch.LongTensor(num_triples).fill_(i))

                all_obj_to_scene.append(torch.LongTensor(num_objs).fill_(i))

                obj_offset += num_objs

            all_objs = torch.cat(all_objs)
            # all_objs_grained = torch.cat(all_objs_grained)
            all_boxes = torch.cat(all_boxes)

            all_obj_to_scene = torch.cat(all_obj_to_scene)

            if len(all_triples) > 0:
                all_triples = torch.cat(all_triples)
                all_triple_to_scene = torch.cat(all_triple_to_scene)
            else:
                return -1

            outputs = {'objs': all_objs,
                    #    'objs_grained': all_objs_grained,
                       'tripltes': all_triples,
                       'boxes': all_boxes,
                       'obj_to_scene': all_obj_to_scene,
                       'triple_to_scene': all_triple_to_scene}

            if len(all_sdfs) > 0:
                outputs['sdfs'] = torch.cat(all_sdfs)
            elif len(all_points) > 0:
                all_points = torch.cat(all_points)
                outputs['points'] = all_points

            if len(all_text_feats) > 0:
                all_text_feats = torch.cat(all_text_feats)
                outputs['text_feats'] = all_text_feats
            if len(all_rel_feats) > 0:
                all_rel_feats = torch.cat(all_rel_feats)
                outputs['rel_feats'] = all_rel_feats
            out[key] = outputs

        return out


class nuScenesLayoutTrain(nuScenesLayout):
    def __init__(self, root, split='train', shuffle_objs=False, use_scene_rels=False, 
                 data_len=None, with_changes=True, scale_func='diag', eval=False, 
                 eval_type='addition', with_CLIP=False, bin_angle=False, seed=True, 
                 recompute_feats=False, recompute_clip=False, dataset='nuscenes', **kwargs):
        
        super().__init__(root, split, shuffle_objs, use_scene_rels, data_len, 
                         with_changes, scale_func, eval, eval_type, with_CLIP, 
                         bin_angle, seed, recompute_feats, recompute_clip, dataset)

class nuScenesLayoutVal(nuScenesLayout):
    def __init__(self, root, split='val', shuffle_objs=False, use_scene_rels=False, 
                 data_len=None, with_changes=True, scale_func='diag', eval=False, 
                 eval_type='addition', with_CLIP=False, bin_angle=False, seed=True, 
                 recompute_feats=False, recompute_clip=False, dataset='nuscenes', **kwargs):
        
        super().__init__(root, split, shuffle_objs, use_scene_rels, data_len, 
                         with_changes, scale_func, eval, eval_type, with_CLIP, 
                         bin_angle, seed, recompute_feats, recompute_clip, dataset)

if __name__ == "__main__":
    dataset = nuScenesLayoutTrain(
        root="/home/alan/AlanLiang/Projects/AlanLiang/CentralScene/data/nuscenes",
        split='val_scans',
        shuffle_objs=True,
        use_SDF=False,
        use_scene_rels=True,
        with_changes=True,
        with_CLIP=True,
        seed=False,
        dataset='nuscenes',
        recompute_clip=False)
    a = dataset[0]

    for x in ['encoder', 'decoder']:
        en_obj = a[x]['objs'].cpu().numpy().astype(np.int32)
        en_triples = a[x]['triples'].cpu().numpy().astype(np.int32)
        #instance
        sub = en_triples[:,0]
        obj = en_triples[:,2]
        #cat
        instance_ids = np.array(sorted(list(set(sub.tolist() + obj.tolist())))) #0-n
        cat_ids = en_obj[instance_ids]
        texts = [dataset.classes_r[cat_id] for cat_id in cat_ids]
        objs = dict(zip(instance_ids.tolist(),texts))
        objs = {str(key): value for key, value in objs.items()}
        for rel in en_triples[:,1]:
            if rel == 0:
                txt = 'in'
                txt_list.append(txt)
                continue
            txt = dataset.relationships_dict_r[rel]
            txt_list.append(txt)
        txt_list = np.array(txt_list)
        rel_list = np.vstack((sub,obj,en_triples[:,1],txt_list)).transpose()
        print(a['scan_id'])