# -*- coding: utf-8 -*-
# @Time    : 2020/9/18 11:33
# @Author  : Hui Wang
# @Email   : hui.wang@ruc.edu.cn

"""
SASRec
################################################

Reference:
    Wang-Cheng Kang et al. "Self-Attentive Sequential Recommendation." in ICDM 2018.

Reference:
    https://github.com/kang205/SASRec

"""

import torch
from torch import nn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder
from recbole.model.loss import BPRLoss


def mask_correlated_samples(batch_size):
    N = 2 * batch_size
    mask = torch.ones((N, N), dtype=bool)
    mask = mask.fill_diagonal_(0)
    for i in range(batch_size):
        mask[i, batch_size + i] = 0
        mask[batch_size + i, i] = 0
    return mask


def info_nce(z_i, z_j, temp, batch_size, sim='dot'):
    """
    We do not sample negative examples explicitly.
    Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
    """
    N = 2 * batch_size

    z = torch.cat((z_i, z_j), dim=0)

    if sim == 'cos':
        sim = nn.functional.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temp
    elif sim == 'dot':
        sim = torch.mm(z, z.T) / temp

    sim_i_j = torch.diag(sim, batch_size)
    sim_j_i = torch.diag(sim, -batch_size)

    positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
    mask = mask_correlated_samples(batch_size)
    negative_samples = sim[mask].reshape(N, -1)

    labels = torch.zeros(N).to(positive_samples.device).long()
    logits = torch.cat((positive_samples, negative_samples), dim=1)
    return logits, labels


class IPSRecS(SequentialRecommender):
    r"""
    SASRec is the first sequential recommender based on self-attentive mechanism.

    NOTE:
        In the author's implementation, the Point-Wise Feed-Forward Network (PFFN) is implemented
        by CNN with 1x1 kernel. In this implementation, we follows the original BERT implementation
        using Fully Connected Layer to implement the PFFN.
    """

    def __init__(self, config, dataset):
        super(IPSRecS, self).__init__(config, dataset)

        # load parameters info
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']

        self.initializer_range = config['initializer_range']
        self.loss_type = config['loss_type']

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.seller_embedding = nn.Embedding(self.n_sellers, self.hidden_size, padding_idx=0)

        # self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)

        self.position_embedding_item = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.position_embedding_seller = nn.Embedding(self.max_seq_length, self.hidden_size)

        self.segment_embedding = nn.Embedding(2, self.hidden_size)

        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_attention_mask(self, item_seq, seller_seq):
        """Generate left-to-right uni-directional attention mask for multi-head attention."""
        attention_mask_item = (item_seq > 0).long()
        attention_mask_seller = (seller_seq > 0).long()
        attention_mask = torch.cat((attention_mask_item, attention_mask_seller), dim=1)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        # mask for left-to-right unidirectional
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long().to(item_seq.device)

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def get_attention_mask_item(self, item_seq):
        """Generate left-to-right uni-directional attention mask for multi-head attention."""
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        # mask for left-to-right unidirectional
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long().to(item_seq.device)

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def get_attention_mask_seller(self, seller_seq):
        """Generate left-to-right uni-directional attention mask for multi-head attention."""
        attention_mask = (seller_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        # mask for left-to-right unidirectional
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long().to(seller_seq.device)

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(self, item_seq, seller_seq, item_seq_len):
        position_ids_item = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids_item = position_ids_item.unsqueeze(0).expand_as(item_seq)
        # position_embedding_item = self.position_embedding(position_ids_item)
        position_embedding_item = self.position_embedding_item(position_ids_item)

        position_ids_seller = torch.arange(seller_seq.size(1), dtype=torch.long, device=seller_seq.device)
        position_ids_seller = position_ids_seller.unsqueeze(0).expand_as(seller_seq)
        # position_embedding_seller = self.position_embedding(position_ids_seller)
        position_embedding_seller = self.position_embedding_seller(position_ids_seller)

        segment_ids_item = torch.zeros(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        segment_ids_item = segment_ids_item.unsqueeze(0).expand_as(item_seq)
        segment_embedding_item = self.segment_embedding(segment_ids_item)

        segment_ids_seller = torch.ones(seller_seq.size(1), dtype=torch.long, device=seller_seq.device)
        segment_ids_seller = segment_ids_seller.unsqueeze(0).expand_as(seller_seq)
        segment_embedding_seller = self.segment_embedding(segment_ids_seller)

        item_emb = self.item_embedding(item_seq)
        input_emb_item = item_emb + position_embedding_item + segment_embedding_item

        seller_emb = self.seller_embedding(seller_seq)
        input_emb_seller = seller_emb + position_embedding_seller + segment_embedding_seller

        input_emb = torch.cat((input_emb_item, input_emb_seller), dim=1)

        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq, seller_seq)

        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]
        output1 = self.gather_indexes(output, (item_seq_len - 1))
        output2 = self.gather_indexes(output, (item_seq_len + 49))

        return output1, output2

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        seller_seq = interaction[self.SELLER_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        # seq_output = self.forward(item_seq, seller_seq, item_seq_len)
        seq_output1, seq_output2 = self.forward(item_seq, seller_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        nce_logits, nce_labels = info_nce(seq_output1, seq_output2, temp=1,
                                          batch_size=seq_output1.shape[0], sim='dot')
        aug_nce_fct = nn.CrossEntropyLoss()
        if self.loss_type == 'BPR':
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output1 * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output1 * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            # test_seller_emb = self.seller_embedding.weight
            # test_item_emb = test_item_emb + test_seller_emb
            logits = torch.matmul(seq_output1, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items) + 0.1 * aug_nce_fct(nce_logits, nce_labels)
            return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        seller_seq = interaction[self.SELLER_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output, _ = self.forward(item_seq, seller_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        # test_seller = interaction[self.SELLER_ID]
        # test_seller_emb = self.seller_embedding(test_seller)
        # test_item_emb = test_item_emb + test_seller_emb
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        seller_seq = interaction[self.SELLER_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output, _ = self.forward(item_seq, seller_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        # test_sellers_emb = self.seller_embedding.weight
        # test_items_emb = test_items_emb + test_sellers_emb
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores
