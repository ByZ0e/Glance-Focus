import torch
import random
from torch.utils.data import DataLoader, Dataset
import numpy as np
from torch.utils.data.dataloader import default_collate
from torch.utils.data import Dataset
import torch.nn.functional as F
from collections import defaultdict
import h5py
import json
from pathlib import Path
import pdb
from random import sample
import pandas as pd
from utils.basic_utils import *
from utils.span_utils import *

OPEN_ENDED_QA = ["frameqa", "count", "msrvtt_qa", "msvd_qa", "ivqa", "nextqa_oe"]
MULTI_CHOICE_QA = ["star", "action", "transition", "nextqa_mc"]

def trans_results(results, save_file):
    question_types = ['Interaction', 'Sequence', 'Prediction', 'Feasibility']
    submission = {t: [] for t in question_types}
    q_ids = []
    for result in results:
        q_id = result['question_id']
        if q_id in q_ids:
            continue
        q_ids.append(q_id)
        q_type = result['question_id'].split('_')[0]
        submission[q_type].append({'question_id': result['question_id'], 'answer': result['answer']})
    save_json(submission, save_file, save_pretty=False)


def repeat_tensor_rows(raw_tensor, row_repeats):
    """ repeat raw_tensor[i] row_repeats[i] times.
    Args:
        raw_tensor: (B, *)
        row_repeats: list(int), len(row_repeats) == len(raw_tensor)
    """
    # print('=======')
    # print(raw_tensor)
    assert len(raw_tensor) == len(raw_tensor), "Has to be the same length"
    if sum(row_repeats) == len(row_repeats):
        return raw_tensor
    else:
        return flat_list_of_lists([[raw_tensor[i] for j in range(r)] for i, r in enumerate(row_repeats)])


def chunk_list(examples, chunk_size=2, pad_to_divisible=True):
    n_examples = len(examples)
    remainder = n_examples % chunk_size
    if pad_to_divisible and remainder > 0:
        n_pad = chunk_size - remainder
        pad = random.choices(examples, k=n_pad)  # with replacement
        examples = examples + pad
        n_examples = len(examples)
        remainder = 0
    chunked_examples = []
    n_chunks = int(n_examples / chunk_size)
    n_chunks = n_chunks + 1 if remainder > 0 else n_chunks
    for i in range(n_chunks):
        chunked_examples.append(examples[i*chunk_size: (i+1)*chunk_size])
    return chunked_examples


def mk_input_group(key_grouped_examples, max_n_example_per_group=2, is_train=True,
                   example_unique_key=None):
    """ Re-organize examples into groups. Each input group will have a single image paired
    with X (X=max_n_example_per_img) examples. Images with total #examples > X will be
    split into multiple groups. In the case a group has < X examples, we will copy
    the examples to make the group has X examples.
    Args:
        key_grouped_examples: dict, each key is image/video id,
            each value is a list(example) associated with this image/video
        max_n_example_per_group: int, pair max #examples with each image/video.
           Note that each image can have multiple groups.
        is_train: bool, if True, copy the examples to make sure each input
            group has max_n_example_per_group examples.
        example_unique_key: str, used to make sure no inputs are discarded by matching
            the input and output ids specified by `example_unique_key`
    """
    input_groups = []  # each element is (id, list(example))
    for k, examples in key_grouped_examples.items():
        chunked_examples = chunk_list(examples,
                                      chunk_size=max_n_example_per_group,
                                      pad_to_divisible=is_train)
        for c in chunked_examples:
            input_groups.append((k, c))

    if example_unique_key is not None:
        print(f"Using example_unique_key {example_unique_key} to check whether input and output ids m")
        # sanity check: make sure we did not discard any input example by accident.
        input_question_ids = flat_list_of_lists(
            [[sub_e[example_unique_key] for sub_e in e] for e in key_grouped_examples.values()])
        output_question_ids = flat_list_of_lists(
            [[sub_e[example_unique_key] for sub_e in e[1]] for e in input_groups])
        assert set(input_question_ids) == set(output_question_ids), "You are missing "
    return input_groups


class VideoQADataset(Dataset):
    """ This should work for both train and test (where labels are not available).
    task_type: str, multiple-choice QA or opened-ended QA,
    anno_path: str, datalist file,
    app_feature_h5: str, appearance feature file,
    str2num_file: str, video index mapping file,
    event_anno_file: str, event annotation file,
    mapping_file: str, action index mapping file,
    max_feats: int, maximum visual input,
    num_queries: int, maximum event queries,
    ans2label: str, answer vocabulary file,
    is_train: bool,
    return_label: bool, whether return label in __getitem__
    """

    def __init__(self, task_type, anno_path, app_feature_h5, str2num_file, event_anno_file, action_mapping_file, max_feats,
                 num_queries=10, ans2label=None, is_train=True, return_label=True):
        self.task_type = task_type
        self.is_train = is_train
        self.return_label = return_label
        self.ans2label = ans2label
        self.max_windows = num_queries
        self.anno_path = anno_path
        self.app_feature_h5 = app_feature_h5
        self.max_feats = max_feats
        self.action_mapping_file = action_mapping_file
        self.datalist = self.make_datalist()
        self.qid2data = {d["question_id"]: d for group in self.datalist for d in group[1]}
        self.features = torch.load(app_feature_h5)

        # uncomment for loading features from h5 file...
        # print('loading appearance feature from %s' % (app_feature_h5))
        # with h5py.File(app_feature_h5, 'r') as app_features_file:
        #     app_video_ids = app_features_file['ids'][()]
        # # self.app_feat_id_to_index = {str(id): i for i, id in enumerate(app_video_ids)}
        # self.app_feat_id_to_index = {id.decode(encoding='utf-8'): i for i, id in enumerate(app_video_ids)}

        with open(str2num_file, 'rb') as f:  # TODO: PATH
            self.strID2NumID = json.load(f)
        with open(event_anno_file, 'rb') as f:  # TODO: PATH
            self.event_anno = json.load(f)

    def map_action(self):
        file_ptr = open(self.action_mapping_file, 'r')
        actions = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        actions_list = [a.split(" ", 1)[1] for a in actions]
        return actions_list

    def make_datalist(self):
        raw_datalist = load_jsonl(self.anno_path)
        print(f"Loaded data size {len(raw_datalist)}")

        datalist = []
        qid = 0
        for raw_d in raw_datalist:
            d = dict(
                question=raw_d["question"],
                vid_id=raw_d["gif_name"] if "gif_name" in raw_d else raw_d["video_id"],
                answer=raw_d["answer"],  # int or str
                question_id=raw_d['question_id'],  # be careful, it is not unique across splits,
                options=raw_d["options"]
            )
            qid += 1
            datalist.append(d)
        print(f"datalist {len(datalist)}")

        grouped = defaultdict(list)  # examples grouped by image/video id
        for d in datalist:
            grouped[d["question_id"]].append(d)
        print(f"grouped {len(grouped)}")

        # each group has a single image with multiple questions
        group_datalist = mk_input_group(
            grouped,
            max_n_example_per_group=1,
            is_train=self.is_train
        )
        print(f"group_datalist {len(group_datalist)}")
        return group_datalist

    def get_span_labels(self, windows, ctx_l):
        """
        windows: list([st, ed]) in seconds. E.g. [[26, 36]], corresponding st_ed clip_indices [[13, 17]] (inclusive)
            Note a maximum of `self.max_windows` windows are used.
        returns Tensor of shape (#windows, 2), each row is [center, width] normalized by video length
        """
        windows = torch.Tensor(windows) / ctx_l  # normalized windows in xx
        return windows

    def _get_single_example(self, data):
        example = dict(
            q_str=data["question"],
            question_id=data["question_id"],
            label=data["answer"]
        )
        if self.task_type in MULTI_CHOICE_QA:
            example["options_str_list"] = data["options"]
        elif self.task_type in OPEN_ENDED_QA:
            if self.return_label:
                example["label"] = self.ans2label[example["label"]]  # answer index
        if not self.return_label:
            example["label"] = None
        return example

    def __getitem__(self, index):
        # skip error videos:
        num_retries = 3
        for _ in range(num_retries):
            vid_id, examples = self.datalist[index]  # one video with multiple examples

            # using S3D feature
            video = self.features[vid_id]

            # using C3D appearance feature
            # app_index = self.app_feat_id_to_index[str(self.strID2NumID[vid_id])]
            # with h5py.File(self.app_feature_h5, 'r') as f_app:
            #     appearance_feat = f_app['resnet_features'][app_index]  # (8, 16, 2048)
            # video = torch.from_numpy(appearance_feat).reshape(-1, 2048)

            # using CLIP feature
            # app_index = self.app_feat_id_to_index[vid_id]
            # with h5py.File(self.app_feature_h5, 'r') as f_app:
            #     appearance_feat = f_app['features'][app_index]  # (32, 17, 512)
            # video = torch.from_numpy(appearance_feat).reshape(-1, 512)

            if len(video) < self.max_feats:
                video = video[: self.max_feats]
                if len(video) < self.max_feats:
                    video = torch.cat(
                        [video, torch.zeros(self.max_feats - len(video), video.shape[1])]
                    )
            else:
                sampled = []
                for j in range(self.max_feats):
                    sampled.append(video[(j * len(video)) // self.max_feats])
                video = torch.stack(sampled)
            appearance_feat = video

            examples = [self._get_single_example(e) for e in examples]
            event_anno = self.event_anno[vid_id]
            duration = event_anno['duration']
            events = event_anno['actions']
            span_list, hoi_list = [], []
            for e in events:
                span_list.append(e[-2:])
                hoi_list.append(e[0])
            if len(events) > self.max_windows:
                l = [i for i in range(len(events))]
                sample_idx = sample(l, self.max_windows)
                span_list = [span_list[idx] for idx in sample_idx]
                hoi_list = [hoi_list[idx] for idx in sample_idx]
            span = self.get_span_labels(span_list, duration)
            hoi = torch.Tensor(hoi_list).long()
            return dict(
                vid=torch.Tensor(appearance_feat),
                examples=examples,
                n_examples=len(examples),  # used to create image feature copies.
                span=span,
                hoi=hoi
            )
        else:
            raise RuntimeError(f"Failed to fetch video after {num_retries} retries.")

    def __len__(self):
        return len(self.datalist)


class VideoQACollator(object):
    def __init__(self, task_type='star', n_options=4):
        self.task_type = task_type
        self.n_options = n_options
        if self.task_type == 'nextqa_mc':
            self.n_options = 5

    def collate_batch(self, batch):
        visual_inputs = [d["vid"] for d in batch]  # <list> (B, dict)
        # group data
        text_examples = flat_list_of_lists([d["examples"] for d in batch])
        n_examples_list = [d["n_examples"] for d in batch]  # (B, )
        # group elements data
        # directly concatenate question and option as a single seq.
        if self.task_type in MULTI_CHOICE_QA:
            text_str_list = flat_list_of_lists(
                [[d["q_str"] + " " + d["options_str_list"][i] for i in range(self.n_options)]
                 for d in text_examples]
            )
        else:
            text_str_list = [d["q_str"] for d in text_examples]  # (B, )

        labels = default_collate([int(d["label"]) for d in text_examples]) \
            if text_examples[0]["label"] is not None else None  # (B, #ans)
        question_ids = [d["question_id"] for d in text_examples]
        span_lst = [d["span"] for d in batch]
        hoi_lst = [d["hoi"] for d in batch]
        return dict(
            visual_inputs=visual_inputs,  # [feat1, feat2, ....]
            text_str_list=text_str_list,  # [q+a1, q+a2, q+a3, q+a4, q2+a1, ...]
            question_ids=question_ids,  # ['Sequence_T5_5496']
            labels=labels,  # [tensor([0/1/2/3)]
            n_examples_list=n_examples_list,  # used to create image feature copies.
            span_lst=span_lst,
            hoi_lst=hoi_lst
        )
