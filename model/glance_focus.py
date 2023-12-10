import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch.nn.utils.weight_norm import weight_norm
from utils.span_utils import *


class GF(nn.Module):
    """ This is the 2-stage Transformer Encoder-Decoder module """
    def __init__(
        self,
        transformer,
        num_queries,
        feature_dim,
        output_dim,
        event_pred_dim,
        qa_dataset: Optional[str] = None
    ):
        """Initializes the model.

        Args:
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of queries
            feature_dim: input feature dimensions
            output_dim: output dimensions
            event_pred_dim: number of event classes
            qa_dataset: If not None, train a QA head for the target dataset
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.output_dim = output_dim
        hidden_dim = transformer.d_model
        span_pred_dim = 2
        self.span_embed = MLP(hidden_dim, hidden_dim, span_pred_dim, 3)
        self.event_class_embed = nn.Linear(hidden_dim, event_pred_dim)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = FCNet([feature_dim, hidden_dim], if_weight_norm=True)
        self.position_embeddings = nn.Embedding(num_queries, hidden_dim)
        self.qa_embed = nn.Embedding(1, hidden_dim)
        self.qa_dataset = qa_dataset
        if qa_dataset is not None:
            if qa_dataset == 'star' or qa_dataset == 'nextqa':
                self.answer_head = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.ReLU(True),
                    nn.Linear(hidden_dim * 2, 1)
                )
            else:
                self.answer_head = nn.Linear(hidden_dim, self.output_dim)  # 961, answer_set

    def forward(self, src, mask, captions, encode_and_save=True, memory_cache=None, query_type=None, glance=None):
        """The forward expects a NestedTensor, which consists of:
           - samples.tensor: batched images, of shape [batch_size x N x H x W]
           - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

        It returns a dict with the following elements:
           Glancing Stage:
           - "memory_prompt":   the predicted memory set.
                                Shape= [batch_size x num_queries x D]
           - "pred_logits":     the event classification logits.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
           - "pred_spans":      the normalized timestamp.
                                Shape= [batch_size x num_queries x (c, w)]

           Focusing Stage:
           - "pred_answer":     the qa classification logits.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
        """

        if encode_and_save:
            # encoding...
            assert memory_cache is None
            query_embed = self.query_embed.weight
            if self.qa_dataset is not None:  # add special qa embeddings to query embeddings
                query_embed = torch.cat([query_embed, self.qa_embed.weight], 0)
            if not glance:
                # The visual input of focusing stage is [frame, memory prompts]
                frame_features, memory_prompt = src
                src = self.input_proj(frame_features)
                # pdb.set_trace()
                position_ids = torch.arange(self.num_queries, dtype=torch.long, device=memory_prompt.device).unsqueeze(0)
                position_embeddings = self.position_embeddings(position_ids)
                src = torch.cat([src, memory_prompt+position_embeddings], dim=1)
            else:
                # The visual input of glancing stage is frame only
                src = self.input_proj(src)
            memory_cache = self.transformer(
                src,
                mask,
                query_embed,
                None,
                captions,
                encode_and_save=True,
                text_memory=None,
                img_memory=None,
                text_attention_mask=None,
                glance=glance
            )
            return memory_cache

        else:
            # decoding...
            assert memory_cache is not None
            if glance:
                hs = self.transformer(
                    mask=memory_cache["mask"],
                    query_embed=memory_cache["query_embed"],
                    pos_embed=memory_cache["pos_embed"],
                    encode_and_save=False,
                    text_memory=None,  # glancing stage has no text memory
                    img_memory=memory_cache["img_memory"],
                    text_attention_mask=None,
                    glance=glance
                )
            else:
                # pdb.set_trace()
                hs = self.transformer(
                    mask=memory_cache["mask"],
                    query_embed=memory_cache["query_embed"],
                    pos_embed=memory_cache["pos_embed"],
                    encode_and_save=False,
                    text_memory=memory_cache["text_memory_resized"],
                    img_memory=memory_cache["img_memory"],
                    text_attention_mask=memory_cache["text_attention_mask"],
                    glance=glance
                )  # (6, 11, 512, 16) num_layer, num_query+1, D, B
            out = {}
            if 'qa' in query_type:  # get qa query at focusing stage
                answer_embeds = hs[0, :, -1]  # [batch, num_hid]
                out["pred_answer"] = self.answer_head(answer_embeds)
                if self.qa_dataset == "star":
                    out["pred_answer"] = self.answer_head(answer_embeds).view(-1, 4)
                if self.qa_dataset == "nextqa":
                    out["pred_answer"] = self.answer_head(answer_embeds).view(-1, 5)
            if 'event' in query_type:  # get memory query at glancing stage
                hs = hs[:, :, :-1]  # [num_layers, batch, num_queries, num_hid] (w/o qa query)
                outputs_class = self.event_class_embed(hs)
                outputs_coord = self.span_embed(hs).sigmoid()  # (#layers, bsz, #queries, 2)
                out.update(
                    {
                        "pred_logits": outputs_class[-1],
                        'pred_spans': outputs_coord[-1],
                        "memory_prompt": hs[-1]
                    }
                )

            return out


class SetCriterion_UNS(nn.Module):
    """ This class computes the unsupervised loss for Glance-Focus.
        - QA Classification Loss
        - Global Diversity Loss:
            Generalized Temporal IoU Loss
            Semantic Diversity Loss
        - Indivisual Certainty Loss
    """

    def __init__(self, losses, weight_dict):
        """ Create the criterion.
        Parameters:
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
        """
        super().__init__()
        self.losses = losses
        self.weight_dict = weight_dict

    def loss_qa(self, outputs, targets):
        """QA Classification Loss"""
        assert 'pred_answer' in outputs
        src_logits = outputs['pred_answer']  # (batch_size, #classes)
        target_classes = targets['qa_labels']
        loss_ce = F.cross_entropy(src_logits, target_classes)
        losses = {'loss_qa': loss_ce}
        return losses

    def loss_giou(self, outputs, targets):
        """Generalized Temporal IoU Loss"""
        assert 'pred_spans' in outputs
        src_spans = outputs['pred_spans']  # (#spans, max_v_l * 2)
        # pdb.set_trace()
        loss_giou = []
        for b, spans in enumerate(src_spans):
            gious = torch.triu(generalized_temporal_iou(span_cxw_to_xx(spans), span_cxw_to_xx(spans)), diagonal=1)
            loss_giou.append(gious.mean())
        losses = {}
        losses['loss_giou'] = torch.stack(loss_giou).mean()
        return losses

    def loss_cls(self, outputs, targets):
        """Semantic Diversity Loss"""
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']  # (batch_size, #queries, #classes)
        diversity_loss = []
        # pdb.set_trace()
        for b, logits in enumerate(src_logits):
            softmax_out = nn.Softmax(dim=-1)(logits)
            msoftmax = softmax_out.mean(dim=0)
            div = torch.sum(msoftmax * torch.log(msoftmax + 1e-6))
            diversity_loss.append(div)
        losses = {'loss_cls': torch.stack(diversity_loss).mean()}
        return losses

    def loss_cert(self, outputs, targets):
        """Individual Certainty Loss"""
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']  # (batch_size, #queries, #classes)
        certainty_loss = []
        # pdb.set_trace()
        for b, logits in enumerate(src_logits):
            softmax_out = nn.Softmax(dim=-1)(logits)
            cert = -torch.sum(softmax_out * torch.log(softmax_out + 1e-6), dim=-1).mean()
            certainty_loss.append(cert)
        losses = {'loss_cert': torch.stack(certainty_loss).mean()}
        return losses

    def get_loss(self, loss, outputs, targets, **kwargs):
        loss_map = {
            "qa": self.loss_qa,
            "giou": self.loss_giou,
            "cls": self.loss_cls,
            "cert": self.loss_cert
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets))

        return losses


class SetCriterion_SUP(nn.Module):
    """ This class computes the supervised loss for Glance-Focus.
        The process happens in two steps:
            1) we compute hungarian assignment between ground truth events and the outputs of the model
            2) we supervise each pair of matched ground-truth / prediction (supervise class and timestamp)
    """

    def __init__(self, matcher, losses, weight_dict, eos_coef, event_pred_dim):
        """ Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
        """
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses

        # classification
        empty_weight = torch.ones(event_pred_dim)
        empty_weight[-1] = eos_coef  # lower weight for background
        self.register_buffer('empty_weight', empty_weight)

    def loss_qa(self, outputs, targets, indices):
        """QA Classification Loss"""
        assert 'pred_answer' in outputs
        src_logits = outputs['pred_answer']  # (batch_size, #classes)
        target_classes = targets['qa_labels']
        loss_ce = F.cross_entropy(src_logits, target_classes)
        losses = {'loss_qa': loss_ce}
        return losses

    def loss_l1(self, outputs, targets, indices):
        """Compute the L1 regression loss
           targets dicts must contain the key "spans" containing a tensor of dim [nb_tgt_spans, 2]
           The target spans are expected in format (center, width), normalized by the event length.
        """
        assert 'pred_spans' in outputs
        targets = targets["event_labels"]
        idx = self._get_src_permutation_idx(indices)
        src_spans = outputs['pred_spans'][idx]
        tgt_spans = torch.cat([t['spans'][i] for t, (_, i) in zip(targets, indices)], dim=0)  # (#spans, 2)
        loss_span = F.l1_loss(src_spans, tgt_spans, reduction='none')
        losses = {}
        losses['loss_l1'] = loss_span.mean()
        return losses

    def loss_cls(self, outputs, targets, indices):
        """Event Classification loss"""
        # TODO add foreground and background classifier.  use all non-matched as background.
        assert 'pred_logits' in outputs
        targets = targets["event_labels"]
        # idx is a tuple of two 1D tensors (batch_idx, src_idx), of the same length == #events in batch
        idx = self._get_src_permutation_idx(indices)
        src_logits = outputs['pred_logits'][idx]  # (batch_size, #queries, #classes)
        target_classes = torch.cat([t['hois'][i] for t, (_, i) in zip(targets, indices)], dim=0)  # (#spans, C)
        loss_ce = F.cross_entropy(src_logits, target_classes, self.empty_weight, reduction="none")
        losses = {'loss_cls': loss_ce.mean()}
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx  # two 1D tensors of the same length

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, **kwargs):
        loss_map = {
            "qa": self.loss_qa,
            "l1": self.loss_l1,
            "cls": self.loss_cls
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        # list(tuples), each tuple is (pred_span_indices, tgt_span_indices)
        indices = self.matcher(outputs_without_aux, targets)

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices))
        return losses


class FCNet(nn.Module):
    """Simple class for non-linear fully connect network
    """
    def __init__(self, dims, if_weight_norm=False):
        super(FCNet, self).__init__()

        layers = []
        for i in range(len(dims)-2):
            in_dim = dims[i]
            out_dim = dims[i+1]
            if if_weight_norm:
                layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            else:
                layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())

        if if_weight_norm:
            layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        else:
            layers.append(nn.Linear(dims[-2], dims[-1]))
        layers.append(nn.ReLU())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


if __name__ == '__main__':
    test_spans1 = torch.Tensor([[0, 0.2], [0.5, 1.0]])
    test_spans2 = torch.Tensor([[0.3, 0.5], [0., 1.0]])
    print(test_spans1[:, 1])
    print(test_spans2[:, 0])
    assert (test_spans1[:, 1] >= test_spans1[:, 0]).all()
    assert (test_spans2[:, 1] >= test_spans2[:, 0]).all()
    print(generalized_temporal_iou(test_spans1, test_spans2))
