import json
import pdb

import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import h5py


def span_xx_to_cxw(xx_spans):
    """
    Args:
        xx_spans: tensor, (#windows, 2) or (..., 2), each row is a window of format (st, ed)

    Returns:
        cxw_spans: tensor, (#windows, 2), each row is a window of format (center=(st+ed)/2, width=(ed-st))
    >>> spans = torch.Tensor([[0, 1], [0.2, 0.4]])
    >>> span_xx_to_cxw(spans)
    tensor([[0.5000, 1.0000],
        [0.3000, 0.2000]])
    >>> spans = torch.Tensor([[[0, 1], [0.2, 0.4]]])
    >>> span_xx_to_cxw(spans)
    tensor([[[0.5000, 1.0000],
         [0.3000, 0.2000]]])
    """
    center = xx_spans.sum(-1) * 0.5
    try:
        width = xx_spans[..., 1] - xx_spans[..., 0]
    except:
        print(xx_spans)
    return torch.stack([center, width], dim=-1)


class LEMMA(Dataset):
    def __init__(self, tagged_qas_path, event_anno_path, mode, num_queries, span_loss_type, event_pred_dim, app_feature_h5,
                 motion_feature_h5) -> None:
        super().__init__()
        with open(tagged_qas_path, 'r') as f:
            self.tagged_qas = json.load(f)
        with open(event_anno_path, 'r') as f:
            self.event_anno = json.load(f)
        self.mode = mode

        print('loading appearance feature from %s' % (app_feature_h5))
        with h5py.File(app_feature_h5, 'r') as app_features_file:
            app_video_ids = app_features_file['ids'][()]
        self.app_feat_id_to_index = {str(id): i for i, id in enumerate(app_video_ids)}
        print('loading motion feature from %s' % (motion_feature_h5))
        with h5py.File(motion_feature_h5, 'r') as motion_features_file:
            motion_video_ids = motion_features_file['ids'][()]
        self.motion_feat_id_to_index = {str(id): i for i, id in enumerate(motion_video_ids)}

        self.app_feature_h5 = app_feature_h5
        self.motion_feature_h5 = motion_feature_h5

        self.max_windows = num_queries - 1  # last one for qa query
        self.span_loss_type = span_loss_type
        self.event_pred_dim = event_pred_dim

    def __len__(self):
        return len(self.tagged_qas)

    def __getitem__(self, index):
        item = self.tagged_qas[index]
        question = item['question']
        reasoning_type = item['reasoning_type'].split('$')  # # list of string
        question_encode = item['question_encode']

        question_encode = torch.from_numpy(np.array(question_encode)).long()
        answer_encode = torch.tensor(int(item['answer_encode'])).float()

        video_idx = item['video_id']
        interval = item['interval']
        app_index = self.app_feat_id_to_index[str(video_idx)]
        motion_index = self.motion_feat_id_to_index[str(video_idx)]

        with h5py.File(self.app_feature_h5, 'r') as f_app:
            appearance_feat = f_app['resnet_features'][app_index]  # (8, 16, 2048)
        with h5py.File(self.motion_feature_h5, 'r') as f_motion:
            motion_feat = f_motion['resnext_features'][motion_index]  # (8, 2048)

        appearance_feat = torch.from_numpy(appearance_feat)
        motion_feat = torch.from_numpy(motion_feat)

        self.clip_len = appearance_feat.shape[0] * appearance_feat.shape[1]  # 8 * 16

        # # torch.Size([4]) torch.Size([4, 8, 16, 2048]) torch.Size([4, 8, 2048]) torch.Size([4, 28])
        span = self.event_anno[interval]["span"]
        if len(span) == 0:
            print(interval)
        ctx_l = int(interval.split("|")[-1])
        span = self.get_span_labels(span, ctx_l)
        hoi = self.event_anno[interval]["hoi"]
        # if len(hoi) < self.max_windows:
        #     hoi += [self.event_pred_dim - 1 for i in range(self.max_windows - len(hoi))]  # fill with background index 823
        if len(hoi) > self.max_windows:
            random.shuffle(hoi)
            hoi = hoi[:self.max_windows]
        # pdb.set_trace()
        return answer_encode, appearance_feat, motion_feat, question, question_encode, reasoning_type, span, torch.tensor(hoi)

    def get_span_labels(self, windows, ctx_l):
        """
        windows: list([st, ed]) in seconds. E.g. [[26, 36]], corresponding st_ed clip_indices [[13, 17]] (inclusive)
            Note a maximum of `self.max_windows` windows are used.
        returns Tensor of shape (#windows, 2), each row is [center, width] normalized by video length
        """
        if len(windows) > self.max_windows:
            random.shuffle(windows)
            windows = windows[:self.max_windows]
        # if len(windows) < self.max_windows:
        #     windows += [[0, 0] for i in range(self.max_windows - len(windows))]
        if self.span_loss_type == "l1":
            # windows = torch.Tensor(windows) / (ctx_l * self.clip_len)  # normalized windows in xx
            windows = torch.Tensor(windows) / ctx_l  # normalized windows in xx
            windows = span_xx_to_cxw(windows)  # normalized windows in cxw
        elif self.span_loss_type == "ce":
            windows = torch.Tensor([
                [int(w[0] / self.clip_len), min(int(w[1] / self.clip_len), ctx_l) - 1]
                for w in windows]).long()  # inclusive
        else:
            raise NotImplementedError
        return windows


def collate_func(batch):
    answer_encode_lst, appearance_feat_lst, motion_feat_lst, question_encode_lst, question_lst, span_lst, hoi_lst = [], [], [], [], [], [], []
    reasoning_type_lst = []
    question_len_lst = []

    for i, (
    answer_encode, appearance_feat, motion_feat, question, question_encode, reasoning_type, span, hoi) in enumerate(
            batch):
        question_encode_lst.append(question_encode)
        answer_encode_lst.append(answer_encode)
        reasoning_type_lst.append(reasoning_type)
        appearance_feat_lst.append(appearance_feat)
        motion_feat_lst.append(motion_feat)
        question_len_lst.append(len(question_encode))
        question_lst.append(question)
        span_lst.append(span)  # [[#spans, 2], [...]]
        hoi_lst.append(hoi)

    # pdb.set_trace()
    question_encode_lst = torch.nn.utils.rnn.pad_sequence(question_encode_lst, batch_first=True, padding_value=0)
    answer_encode_lst = torch.tensor(answer_encode_lst)
    appearance_feat_lst = torch.stack(appearance_feat_lst, dim=0)
    motion_feat_lst = torch.stack(motion_feat_lst, dim=0)
    # span_lst = torch.stack(span_lst, dim=0)
    # hoi_lst = torch.stack(hoi_lst, dim=0)
    # #
    return answer_encode_lst, appearance_feat_lst, motion_feat_lst, question_lst, question_encode_lst, question_len_lst, reasoning_type_lst, span_lst, hoi_lst


if __name__ == '__main__':
    dataset = LEMMA('/home/leiting/scratch/lemma_simple_model/data/formatted_test_qas_encode.json',
                    mode='train',
                    app_feature_h5='data/hcrn_data/lemma-qa_appearance_feat.h5',
                    motion_feature_h5='data/hcrn_data/lemma-qa_motion_feat.h5')

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_func)
    for i, (
    answer_encode, appearance_feat, motion_feat, question_encode, question_len_lst, reasoning_type_lst) in enumerate(
            dataloader):
        print(i, answer_encode.shape, appearance_feat.shape, motion_feat.shape, question_encode.shape)
        print(len(question_len_lst))
        break
