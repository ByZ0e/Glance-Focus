import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import sys, os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import argparse, time
from dataset.star import VideoQADataset, VideoQACollator, repeat_tensor_rows, trans_results
from model.transformer_gf import build_transformer
from model.glance_focus import GF, SetCriterion_SUP
from model.matcher import build_matcher

OPEN_ENDED_QA = ["frameqa", "count", "msrvtt_qa", "msvd_qa", "ivqa", "nextqa_oe"]
MULTI_CHOICE_QA = ["star", "action", "transition", "nextqa_mc"]


def parse_args():
    parser = argparse.ArgumentParser()

    # * Training Parameters
    parser.add_argument("--basedir", type=str, default='expm/star',
                        help='where to store ckpts and logs')
    parser.add_argument("--name", type=str, default='gf_logs',
                        help='where to store ckpts and logs')
    parser.add_argument('--device_id', default=0, type=int, help='device id')
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--nepoch", type=int, default=10, help='num of total epoches')
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--i_val", type=int, default=300,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_test", type=int, default=300,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_print", type=int, default=100,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_weight", type=int, default=1000,
                        help='frequency of weight ckpt saving')
    parser.add_argument('--test_only', default=0, type=int)
    parser.add_argument('--reload_model_path', default='', type=str, help='model_path')
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_queries', type=int, default=10)
    parser.add_argument('--event_pred_dim', type=int, default=50)
    parser.add_argument('--max_feats', type=int, default=80)

    # * Dataset
    parser.add_argument('--qa_dataset', default='star', type=str, help='qa dataset')
    parser.add_argument('--task_type', default='star', type=str, help='task type, multi-choice or open-ended')
    parser.add_argument('--num_options', type=int, default=4, help='number of options for multi-choice QA')
    parser.add_argument('--output_dim', type=int, default=1, help='vocabulary scale for open-ended QA')
    parser.add_argument("--base_data_dir", type=str, default='', help='base data directory')
    parser.add_argument("--train_data_file_path", type=str, default='{}/txt_db/train.jsonl')
    parser.add_argument("--test_data_file_path", type=str, default='{}/txt_db/test.jsonl')
    parser.add_argument("--val_data_file_path", type=str, default='{}/txt_db/val.jsonl')
    parser.add_argument('--event_anno_path', type=str, default='{}/txt_db/events.json')
    parser.add_argument('--action_mapping_file', type=str, default='{}/txt_db/action_mapping.txt')
    parser.add_argument('--app_feat_path', type=str, default='{}/vis_db/s3d.pth')
    parser.add_argument('--feature_dim', type=int, default=1024)
    parser.add_argument('--str2num_file', type=str, default='{}/vis_db/strID2numID.json')

    # * Matcher
    parser.add_argument('--set_cost_l1', default=1, type=float,
                        help="L1 span coefficient in the matching cost")
    parser.add_argument('--set_cost_cls', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # * Loss coefficients
    parser.add_argument('--losses_type', default=['qa','cls','l1'], type=list)
    parser.add_argument('--qa_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=0.5, type=float)
    parser.add_argument('--l1_loss_coef', default=0.5, type=float)
    args = parser.parse_args()
    return args


def forward_step(batch, args):
    """shared for training and validation. Repeat for multi-choice tasks"""
    if args.task_type in MULTI_CHOICE_QA:
        repeat_counts = [e * args.num_options for e in batch["n_examples_list"]]
        del batch["n_examples_list"]
        batch["visual_inputs"] = torch.stack(repeat_tensor_rows(batch["visual_inputs"], repeat_counts)).to(args.device)
        batch['span_lst'] = repeat_tensor_rows(batch['span_lst'], repeat_counts)
        batch['hoi_lst'] = repeat_tensor_rows(batch['hoi_lst'], repeat_counts)
    batch['labels'] = batch['labels'].to(args.device)
    return batch


def sorter(outputs_event):
    memories = outputs_event['memory_prompt']
    pred_centers = outputs_event["pred_spans"][:, :, 0]
    _, index = pred_centers.sort(dim=1)
    batch_idx = torch.cat([torch.full_like(idx, i) for i, idx in enumerate(index)])
    return memories[batch_idx, index.view(-1)].view(-1, args.num_queries, args.hidden_dim)


def train(args):
    device = args.device

    train_dataset = VideoQADataset(args.task_type, args.train_data_file_path.format(args.base_data_dir),
                                   args.app_feat_path.format(args.base_data_dir),
                                   args.str2num_file.format(args.base_data_dir),
                                   args.event_anno_path.format(args.base_data_dir),
                                   args.action_mapping_file.format(args.base_data_dir), args.max_feats,
                                   num_queries=args.num_queries, is_train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=VideoQACollator(task_type=args.task_type).collate_batch)

    val_dataset = VideoQADataset(args.task_type, args.val_data_file_path.format(args.base_data_dir),
                                 args.app_feat_path.format(args.base_data_dir),
                                 args.str2num_file.format(args.base_data_dir),
                                 args.event_anno_path.format(args.base_data_dir),
                                 args.action_mapping_file.format(args.base_data_dir), args.max_feats,
                                 num_queries=args.num_queries, is_train=False)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                collate_fn=VideoQACollator(task_type=args.task_type).collate_batch)

    hois = train_dataset.map_action()
    args.event_pred_dim = len(hois) + 1  # add extra one for bg

    transformer = build_transformer(args)
    model = GF(
        transformer,
        num_queries=args.num_queries,
        feature_dim=args.feature_dim,
        output_dim=args.output_dim,
        event_pred_dim=args.event_pred_dim,
        qa_dataset=args.qa_dataset
    ).to(device)

    matcher = build_matcher(args)
    weight_dict = {"loss_qa": args.qa_loss_coef,
                   "loss_l1": args.l1_loss_coef,
                   "loss_cls": args.cls_loss_coef
                   }

    criterion = SetCriterion_SUP(matcher=matcher, losses=args.losses_type, weight_dict=weight_dict,
                                 eos_coef=args.eos_coef, event_pred_dim=args.event_pred_dim)
    criterion.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)


    reload_step = 0
    if args.reload_model_path != '':
        print('reloading model from', args.reload_model_path)
        reload_step = reload(model=model, optimizer=optimizer, path=args.reload_model_path)

    print(
        f"Nb of trainable params:{sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )

    global_step = reload_step
    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
    args.basedir = os.path.join(args.basedir, args.name)
    log_dir = os.path.join(args.basedir, 'events', TIMESTAMP)
    os.makedirs(log_dir)
    with open(os.path.join(log_dir, 'argument.txt'), 'w') as f:
        for key, value in vars(args).items():
            f.write('%s:%s\n' % (key, value))
            print(key, value)

    log_file = open(os.path.join(log_dir, 'log.txt'), 'w')
    writer = SummaryWriter(log_dir=log_dir)

    os.makedirs(os.path.join(args.basedir, 'ckpts_{}'.format(TIMESTAMP)), exist_ok=True)
    pbar = tqdm(total=args.nepoch * len(train_dataloader))
    global_val_acc = 0

    for epoch in range(args.nepoch):
        model.train()
        for b, batch in enumerate(train_dataloader):
            batch = forward_step(batch, args)
            answer_encode = batch['labels']
            B, num_frames, D = batch['visual_inputs'].shape
            frame_features = batch['visual_inputs'].to(device)
            visual_attention_mask = torch.ones(frame_features.shape[:-1], dtype=torch.float).to(device)
            # Glancing Stage
            memory_cache = model(frame_features, visual_attention_mask, None, encode_and_save=True, glance=True)
            outputs_event = model(frame_features, visual_attention_mask, None, encode_and_save=False, glance=True,
                                  memory_cache=memory_cache, query_type='event')
            # Focusing Stage
            text_input = batch['text_str_list']
            memory_prompt = sorter(outputs_event)
            frame_features = (frame_features, memory_prompt)
            visual_attention_mask = torch.ones((B, num_frames+args.num_queries), dtype=torch.float).to(device)
            memory_cache = model(frame_features, visual_attention_mask, text_input, encode_and_save=True,
                                 glance=False)
            outputs_qa = model(frame_features, visual_attention_mask, text_input, encode_and_save=False, glance=False,
                               memory_cache=memory_cache, query_type='qa')

            logits = outputs_qa['pred_answer']
            event_labels = [dict(spans=batch['span_lst'][idx].to(device), hois=batch['hoi_lst'][idx].to(device))
                            for idx in range(B)]
            targets = dict(event_labels=event_labels, qa_labels=answer_encode.long())

            outputs = {}
            outputs.update(outputs_event)
            outputs.update(outputs_qa)
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            pred = torch.argmax(logits, dim=1)
            train_acc = sum(pred == answer_encode) / len(answer_encode)

            writer.add_scalar('train/loss', losses.item(), global_step)
            writer.add_scalar('learning rates', optimizer.param_groups[0]['lr'], global_step)
            writer.add_scalar('train/acc', train_acc, global_step)

            pbar.update(1)
            if global_step % args.i_print == 0:
                print(f"global_step:{global_step}, train_loss:{losses.item()}, train_acc:{train_acc}")
                log_file.write(f'global_step: {global_step}, train_loss: {losses.item()}, train_acc:{train_acc}\n')

            if (global_step) % args.i_val == 0:
                val_loss, val_acc, _ = validate(model, val_dataloader, criterion, args)
                writer.add_scalar('val/loss', val_loss.item(), global_step)
                writer.add_scalar('val/acc', val_acc, global_step)

                log_file.write(f'[VAL]: epoch: {epoch}, global_step: {global_step}\n')
                log_file.write(f'val/loss: {val_loss.item()}, val/acc: {val_acc}\n')

            if (global_step) % args.i_weight == 0 and global_step >= 3000 and val_acc >= global_val_acc:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': losses,
                    'global_step': global_step,
                }, os.path.join(args.basedir, 'ckpts_{}'.format(TIMESTAMP), f"model_{global_step}.tar"))
                global_val_acc = val_acc
            global_step += 1

        log_file.write(f'[TRAIN]: epoch: {epoch}, global_step: {global_step}\n')
        log_file.flush()


def test(args):
    device = args.device
    test_dataset = VideoQADataset(args.task_type, args.test_data_file_path.format(args.base_data_dir),
                                  args.app_feat_path.format(args.base_data_dir),
                                  args.str2num_file.format(args.base_data_dir),
                                  args.event_anno_path.format(args.base_data_dir),
                                  args.action_mapping_file.format(args.base_data_dir), args.max_feats,
                                  num_queries=args.num_queries, is_train=False, return_label=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                 collate_fn=VideoQACollator(task_type=args.task_type).collate_batch)

    transformer = build_transformer(args)
    model = GF(
        transformer,
        num_queries=args.num_queries,
        feature_dim=args.feature_dim,
        output_dim=args.output_dim,
        event_pred_dim=args.event_pred_dim,
        qa_dataset=args.qa_dataset
    ).to(device)

    weight_dict = {"loss_qa": args.qa_loss_coef,
                   "loss_l1": args.l1_loss_coef,
                   "loss_cls": args.cls_loss_coef
                   }
    matcher = build_matcher(args)
    criterion = SetCriterion_SUP(matcher=matcher, losses=args.losses_type, weight_dict=weight_dict,
                                 eos_coef=args.eos_coef, event_pred_dim=args.event_pred_dim)
    criterion.to(device)

    test_loss, test_acc, results = validate(model, test_dataloader, criterion, args)
    TIMESTAMP = args.reload_model_path.split('/')[-2].split('_')[-1]
    if args.qa_dataset == 'star':  # write to submission file
        trans_results(results, os.path.join('/'.join(args.reload_model_path.split('/')[:3]), 'events', TIMESTAMP,
                                            'submission_{}.json'.format(TIMESTAMP)))
    print('TEST ACC:', test_acc)


def validate(model, val_loader, criterion, args):
    model.eval()
    all_acc = 0
    all_loss = 0
    qa_results = []

    pbar = tqdm(total=len(val_loader))

    print('validating...')
    with torch.no_grad():
        for b, batch in enumerate(val_loader):
            if batch['labels'] == None:   # no test gts are provided.
                batch['labels'] = torch.zeros(len(batch['visual_inputs']))
            batch = forward_step(batch, args)
            answer_encode = batch['labels']
            B, num_frames, D = batch['visual_inputs'].shape
            frame_features = batch['visual_inputs'].to(device)
            visual_attention_mask = torch.ones(frame_features.shape[:-1], dtype=torch.float).to(device)
            # Glancing Stage
            memory_cache = model(frame_features, visual_attention_mask, None, encode_and_save=True, glance=True)
            outputs_event = model(frame_features, visual_attention_mask, None, encode_and_save=False, glance=True,
                                  memory_cache=memory_cache, query_type='event')
            # Focusing Stage
            text_input = batch['text_str_list']
            memory_prompt = sorter(outputs_event)
            frame_features = (frame_features, memory_prompt)
            visual_attention_mask = torch.ones((B, num_frames + args.num_queries), dtype=torch.float).to(device)
            memory_cache = model(frame_features, visual_attention_mask, text_input, encode_and_save=True, glance=False)
            outputs_qa = model(frame_features, visual_attention_mask, text_input, encode_and_save=False, glance=False,
                               memory_cache=memory_cache, query_type='qa')

            logits = outputs_qa['pred_answer']
            event_labels = [dict(spans=batch['span_lst'][idx].to(device), hois=batch['hoi_lst'][idx].to(device))
                            for idx in range(B)]
            if answer_encode == None:
                targets = dict(event_labels=event_labels)
            else:
                targets = dict(event_labels=event_labels, qa_labels=answer_encode.long())

            outputs = {}
            outputs.update(outputs_event)
            outputs.update(outputs_qa)
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            all_loss += sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            pred = torch.argmax(logits, dim=1)
            test_acc = sum(pred == answer_encode) / len(answer_encode)
            all_acc += test_acc

            pred_labels = logits.max(dim=-1)[1].data.tolist()
            for qid, pred_label in zip(batch['question_ids'], pred_labels):
                qa_results.append(dict(
                    question_id=qid,
                    answer=pred_label,
                    data=val_loader.dataset.qid2data[qid]
                ))

            pbar.update(1)

    all_loss /= len(val_loader)
    all_acc /= len(val_loader)
    model.train()
    return all_loss, all_acc, qa_results


def reload(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    global_step = checkpoint['global_step']
    return global_step


if __name__ == '__main__':
    args = parse_args()
    device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
    args.device = device
    if args.test_only:
        print('test only!')
        print('loading model from', args.reload_model_path)
        test(args)
    else:
        print('start training...')
        train(args)
