#! /usr/bin/env python3

import torch


class OnlineTripletMiner(torch.nn.Module):
    def __init__(self, margin, normalize_embeddings=True, sample_triplets_type="semi_hard"):
        super().__init__()
        self.normalize_embeddings = normalize_embeddings
        self.sample_triplets_type = sample_triplets_type
        self.margin = margin

    def assert_embeddings_and_labels_are_same_size(self, embeddings, labels):
        assert embeddings.size(0) == labels.size(0), "Number of embeddings must equal number of labels"

    def get_all_triplets_indices(self, labels, ref_labels=None):
        if ref_labels is None:
            ref_labels = labels
        labels1 = labels.unsqueeze(1)
        labels2 = ref_labels.unsqueeze(0)
        matches = (labels1 == labels2).byte()
        diffs = matches ^ 1
        if ref_labels is labels:
            matches -= torch.eye(matches.size(0)).byte().to(labels.device)
        triplets = matches.unsqueeze(2)*diffs.unsqueeze(1)
        a_idx = triplets.nonzero()[:, 0].flatten()
        p_idx = triplets.nonzero()[:, 1].flatten()
        n_idx = triplets.nonzero()[:, 2].flatten()
        keep = labels[a_idx] == 0  # anchors should be only from label=0
        a_idx = a_idx[keep]
        p_idx = p_idx[keep]
        n_idx = n_idx[keep]
        return a_idx, p_idx, n_idx

    def mine(self, embeddings, labels):
        anchor_idx, positive_idx, negative_idx  = self.get_all_triplets_indices(labels)
        anchors, positives, negatives = embeddings[anchor_idx], embeddings[positive_idx], embeddings[negative_idx]
        ap_dist = torch.nn.functional.pairwise_distance(anchors, positives, 2)
        an_dist = torch.nn.functional.pairwise_distance(anchors, negatives, 2)
        triplet_margin = ap_dist - an_dist
        threshold_condition = triplet_margin > -self.margin
        if self.sample_triplets_type == "hard":
            threshold_condition &= an_dist < ap_dist
        elif self.sample_triplets_type == "semihard":
            threshold_condition &= an_dist > ap_dist
        return anchor_idx[threshold_condition], positive_idx[threshold_condition], negative_idx[threshold_condition]

    def output_assertion(self, output):
        """
        Args:
            output: the output of self.mine
        This asserts that the mining function is outputting
        properly formatted indices. The default is to require a tuple representing
        a,p,n indices within a batch of embeddings.
        For example, a tuple of (anchors, positives, negatives) will be
        (torch.tensor, torch.tensor, torch.tensor)
        """
        if len(output) == 3:
            self.num_triplets = len(output[0])
            assert self.num_triplets == len(output[1]) == len(output[2])
        else:
            raise BaseException

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: tensor of size (batch_size, embedding_size)
            labels: tensor of size (batch_size, label_size)
        Does any necessary preprocessing, then does mining, and then checks the
        shape of the mining output before returning it
        """
        sub_labels = labels[:, 1]
        with torch.no_grad():
            self.assert_embeddings_and_labels_are_same_size(embeddings, sub_labels)
            # sub_labels = sub_labels.to(embeddings.device)
            if self.normalize_embeddings:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            mining_output = self.mine(embeddings, sub_labels)
        self.output_assertion(mining_output)
        return mining_output
