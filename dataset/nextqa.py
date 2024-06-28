import os.path

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
import pandas as pd
import collections
import h5py
import glob


def flat_list_of_lists(l):
    """flatten a list of lists [[1,2], [3,4]] to [1,2,3,4]"""
    return [item for sublist in l for item in sublist]


def tokenize(
    seq,
    tokenizer,
    add_special_tokens=True,
    max_length=10,
    dynamic_padding=True,
    truncation=True,
):
    """
    :param seq: sequence of sequences of text
    :param tokenizer: bert_tokenizer
    :return: torch tensor padded up to length max_length of bert tokens
    """
    tokens = tokenizer.batch_encode_plus(
        seq,
        add_special_tokens=add_special_tokens,
        max_length=max_length,
        padding="longest" if dynamic_padding else "max_length",
        truncation=truncation,
    )["input_ids"]
    return torch.tensor(tokens, dtype=torch.long)


class VideoQADataset(Dataset):
    def __init__(
        self,
        csv_path,
        features,
        qmax_words=20,
        amax_words=5,
        bert_tokenizer=None,
        a2id=None,
        ivqa=False,
        max_feats=20,
        mc=0,
        num_vocab=2000
    ):
        """
        :param csv_path: path to a csv containing columns video_id, question, answer
        :param features: dictionary mapping video_id to torch tensor of features
        :param qmax_words: maximum number of words for a question
        :param amax_words: maximum number of words for an answer
        :param bert_tokenizer: BERT tokenizer
        :param a2id: answer to index mapping
        :param ivqa: whether to use iVQA or not
        :param max_feats: maximum frames to sample from a video
        """
        self.data = pd.read_csv(csv_path)
        self.qmax_words = qmax_words
        self.amax_words = amax_words
        self.a2id = a2id
        self.bert_tokenizer = bert_tokenizer
        self.ivqa = ivqa
        self.max_feats = max_feats
        self.num_options = mc
        self.num_vocab = num_vocab
        self.features = features
        # self.feature_h5 = features
        # with h5py.File(self.feature_h5, 'r') as features_file:
        #     video_ids = features_file['ids'][()]
        # self.feat_id_to_index = {str(id): i for i, id in enumerate(video_ids)}
        # video_ids = glob.glob('/data1/bzy/NExT-QA/video/*/*.mp4')
        # self.feat_id_to_index = {str(id).split('/')[-1].replace('.mp4', ''): i for i, id in enumerate(video_ids)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        vid_id = self.data["video_id"].values[index]
        video = self.features[vid_id].float()
        # try:
        #     f_index = self.feat_id_to_index[str(vid_id)]
        #     with h5py.File(self.feature_h5, 'r') as features_file:
        #         video = torch.from_numpy(features_file['feat'][f_index])  # feat for mot_app
        # except:
        #     video = torch.rand(1, 1, 512)
        #     print(str(vid_id))

        if len(video.shape) == 4:
            video = video.reshape(-1, 2048)
        if len(video.shape) == 3:
            video = video.reshape(-1, 512)
        if len(video) <= self.max_feats:
            video = video[: self.max_feats]
            vid_duration = len(video)
            if len(video) < self.max_feats:
                video = torch.cat(
                    [video, torch.zeros(self.max_feats - len(video), video.shape[1])]
                )
        else:
            sampled = []
            for j in range(self.max_feats):
                sampled.append(video[(j * len(video)) // self.max_feats])
            video = torch.stack(sampled)
            vid_duration = len(video)

        type, answer, answer_len = 0, 0, 0
        if self.ivqa:
            answer_txt = collections.Counter(
                [
                    self.data["answer1"].values[index],
                    self.data["answer2"].values[index],
                    self.data["answer3"].values[index],
                    self.data["answer4"].values[index],
                    self.data["answer5"].values[index],
                ]
            )
            answer_id = torch.zeros(len(self.a2id))
            for x in answer_txt:
                if x in self.a2id:
                    answer_id[self.a2id[x]] = answer_txt[x]
            answer_txt = ", ".join(
                [str(x) + "(" + str(answer_txt[x]) + ")" for x in answer_txt]
            )
        elif self.num_options:
            answer_id = int(self.data["answer"][index])
            answer_txt = [self.data["a" + str(i)][index] for i in range(self.num_options)]
            answer = tokenize(
                answer_txt,
                self.bert_tokenizer,
                add_special_tokens=True,
                max_length=self.amax_words,
                dynamic_padding=True,
                truncation=True,
            )
        else:
            answer_txt = self.data["answer"].values[index]
            answer_id = self.a2id.get(
                answer_txt, self.num_vocab - 1
            )  # put an answer_id 'num_vocab-1' if not in top answers, that will be considered wrong during evaluation
        if not self.num_options:
            type = self.data["type"].values[index]

        question_txt = self.data["question"][index]
        question_embd = torch.tensor(
            self.bert_tokenizer.encode(
                question_txt,
                add_special_tokens=True,
                padding="longest",
                max_length=self.qmax_words,
                truncation=True,
            ),
            dtype=torch.long,
        )

        return {
            "video_id": vid_id,
            "video": video,
            "video_len": vid_duration,
            "question": question_embd,
            "question_id": self.data["qid"].values[index],
            "question_txt": [question_txt],
            "type": type,
            "answer_id": answer_id,
            "answer_txt": answer_txt,
            "answer": answer,
        }


def videoqa_collate_fn(batch):
    """
    :param batch: [dataset[i] for i in N]
    :return: tensorized batch with the question and the ans candidates padded to the max length of the batch
    """
    qmax_len = max(len(batch[i]["question"]) for i in range(len(batch)))
    for i in range(len(batch)):
        if len(batch[i]["question"]) < qmax_len:
            batch[i]["question"] = torch.cat(
                [
                    batch[i]["question"],
                    torch.zeros(qmax_len - len(batch[i]["question"]), dtype=torch.long),
                ],
                0,
            )
        if len(batch[i]["answer_txt"]) > 1:
            batch[i]["answer_txt"] = [batch[i]["question_txt"][0] + " [SEP] " + a for a in batch[i]["answer_txt"]]

    if not isinstance(batch[0]["answer"], int):
        amax_len = max(x["answer"].size(1) for x in batch)
        for i in range(len(batch)):
            if batch[i]["answer"].size(1) < amax_len:
                batch[i]["answer"] = torch.cat(
                    [
                        batch[i]["answer"],
                        torch.zeros(
                            (
                                batch[i]["answer"].size(0),
                                amax_len - batch[i]["answer"].size(1),
                            ),
                            dtype=torch.long,
                        ),
                    ],
                    1,
                )

    # return default_collate(batch)
    return dict(
        video_id=[d["video_id"] for d in batch],
        question_ids=[d["question_id"] for d in batch],
        video=[d["video"] for d in batch],
        answer_id=[d["answer_id"] for d in batch],
        answer_txt=flat_list_of_lists([d["answer_txt"] for d in batch])
    )
    # return dict(
    #         video=[d["video"] for d in batch],
    #         answer_id=[d["answer_id"] for d in batch],
    #         question_txt=[d["question_txt"] for d in batch],
    #         answer=[d["answer"] for d in batch]
    #     )


def get_videoqa_loaders(args, features, a2id, bert_tokenizer):
    train_dataset = VideoQADataset(
        csv_path=args.train_csv_path.format(args.base_data_dir),
        features=features,
        # features=os.path.join(features, 'region_16c20b_train.h5'),  # app_mot_val_train
        qmax_words=args.qmax_words,
        amax_words=args.amax_words,
        bert_tokenizer=bert_tokenizer,
        a2id=a2id,
        ivqa=(args.dataset == "ivqa"),
        max_feats=args.max_feats,
        mc=args.num_options,
        num_vocab=args.output_dim
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_thread_reader,
        shuffle=True,
        drop_last=True,
        collate_fn=videoqa_collate_fn,
    )

    test_dataset = VideoQADataset(
        csv_path=args.test_csv_path.format(args.base_data_dir),
        features=features,
        # features=os.path.join(features, 'region_16c20b_test.h5'),  # app_mot_val_test
        qmax_words=args.qmax_words,
        amax_words=args.amax_words,
        bert_tokenizer=bert_tokenizer,
        a2id=a2id,
        ivqa=(args.dataset == "ivqa"),
        max_feats=args.max_feats,
        mc=args.num_options,
        num_vocab=args.output_dim,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
        collate_fn=videoqa_collate_fn,
    )

    val_dataset = VideoQADataset(
        csv_path=args.val_csv_path.format(args.base_data_dir),
        features=features,
        # features=os.path.join(features, 'region_16c20b_val.h5'),  # app_mot_val
        qmax_words=args.qmax_words,
        amax_words=args.amax_words,
        bert_tokenizer=bert_tokenizer,
        a2id=a2id,
        ivqa=(args.dataset == "ivqa"),
        max_feats=args.max_feats,
        mc=args.num_options,
        num_vocab=args.output_dim
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        collate_fn=videoqa_collate_fn,
    )
    return (
        train_dataset,
        train_loader,
        val_dataset,
        val_loader,
        test_dataset,
        test_loader,
    )
