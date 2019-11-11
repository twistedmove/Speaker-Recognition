import torch
import torch.nn.functional as F
from torch import nn


class CustomCrossEntropyLoss(nn.Module):

    __name__ = 'CustomCrossEntropyLoss'

    def __init__(self):
        super(CustomCrossEntropyLoss, self).__init__()

        self.ce = nn.CrossEntropyLoss()

    def forward(self, pr, gt):

        return self.ce(pr, gt)


class AMSoftMaxV1(nn.Module):
    __name__ = 'AMSoftMaxV1'

    def __init__(self,
                 m=0.35,
                 s=30):
        super(AMSoftMaxV1, self).__init__()
        self.m = m
        self.s = s
        self.ce = nn.CrossEntropyLoss()

    def forward(self, pr, gt):
        assert pr.size()[0] == gt.size()[0]

        gt = gt.unsqueeze(-1)
        pr = gt * (pr - self.m) + (1 - gt) * pr
        pr *= self.s
        loss = self.ce(pr, gt.squeeze(-1).long())
        return loss


class GE2ELoss(nn.Module):
    __name__ = 'GE2ELoss'

    def __init__(self, init_w=10.0, init_b=-5.0, loss_method='softmax', device='cuda'):
        '''
        Implementation of the Generalized End-to-End loss defined in https://arxiv.org/abs/1710.10467 [1]
        Accepts an input of size (N, M, D)
            where N is the number of speakers in the batch,
            M is the number of utterances per speaker,
            and D is the dimensionality of the embedding vector (e.g. d-vector)
        Args:
            - init_w (float): defines the initial value of w in Equation (5) of [1]
            - init_b (float): definies the initial value of b in Equation (5) of [1]
        '''
        super(GE2ELoss, self).__init__()
        self.w = nn.Parameter(torch.tensor(init_w).to(device), requires_grad=True)
        self.b = nn.Parameter(torch.tensor(init_b).to(device), requires_grad=True)
        self.device = device
        self.loss_method = loss_method

        assert self.loss_method in ['softmax', 'contrast']

        if self.loss_method == 'softmax':
            self.embed_loss = self.embed_loss_softmax
        if self.loss_method == 'contrast':
            self.embed_loss = self.embed_loss_contrast

    def calc_new_centroids(self, dvecs, centroids, spkr, utt):
        '''
        Calculates the new centroids excluding the reference utterance
        '''
        excl = torch.cat((dvecs[spkr,:utt], dvecs[spkr,utt+1:]))
        excl = torch.mean(excl, 0)
        new_centroids = []
        for i, centroid in enumerate(centroids):
            if i == spkr:
                new_centroids.append(excl)
            else:
                new_centroids.append(centroid)
        return torch.stack(new_centroids)

    def calc_cosine_sim(self, dvecs, centroids):
        '''
        Make the cosine similarity matrix with dims (N,M,N)
        '''
        cos_sim_matrix = []
        for spkr_idx, speaker in enumerate(dvecs):
            cs_row = []
            for utt_idx, utterance in enumerate(speaker):
                new_centroids = self.calc_new_centroids(dvecs, centroids, spkr_idx, utt_idx)
                # vector based cosine similarity for speed
                cs_row.append(torch.clamp(torch.mm(utterance.unsqueeze(1).transpose(0,1), new_centroids.transpose(0,1)) / (torch.norm(utterance) * torch.norm(new_centroids, dim=1)), 1e-6))
            cs_row = torch.cat(cs_row, dim=0)
            cos_sim_matrix.append(cs_row)
        return torch.stack(cos_sim_matrix)

    def embed_loss_softmax(self, dvecs, cos_sim_matrix):
        '''
        Calculates the loss on each embedding $L(e_{ji})$ by taking softmax
        '''
        N, M, _ = dvecs.shape
        L = []
        for j in range(N):
            L_row = []
            for i in range(M):
                L_row.append(-F.log_softmax(cos_sim_matrix[j,i], 0)[j])
            L_row = torch.stack(L_row)
            L.append(L_row)
        return torch.stack(L)

    def embed_loss_contrast(self, dvecs, cos_sim_matrix):
        '''
        Calculates the loss on each embedding $L(e_{ji})$ by contrast loss with closest centroid
        '''
        N, M, _ = dvecs.shape
        L = []
        for j in range(N):
            L_row = []
            for i in range(M):
                centroids_sigmoids = torch.sigmoid(cos_sim_matrix[j,i])
                excl_centroids_sigmoids = torch.cat((centroids_sigmoids[:j], centroids_sigmoids[j+1:]))
                L_row.append(1. - torch.sigmoid(cos_sim_matrix[j,i,j]) + torch.max(excl_centroids_sigmoids))
            L_row = torch.stack(L_row)
            L.append(L_row)
        return torch.stack(L)

    def forward(self, dvecs):
        '''
        Calculates the GE2E loss for an input of dimensions (num_speakers, num_utts_per_speaker, dvec_feats)
        '''
        #Calculate centroids
        centroids = torch.mean(dvecs, 1)

        #Calculate the cosine similarity matrix
        cos_sim_matrix = self.calc_cosine_sim(dvecs, centroids)
        torch.clamp(self.w, 1e-6)
        cos_sim_matrix = cos_sim_matrix * self.w + self.b
        L = self.embed_loss(dvecs, cos_sim_matrix)
        return L.sum()


class HardTripletLoss(nn.Module):

    __name__ = "HardTripletLoss"

    def __init__(self, margin: float = 0.2, squared: bool = False):
        super(HardTripletLoss, self).__init__()

        self.margin = margin
        self.squared = squared

    def forward(self, embeddings, targets):

        return batch_hard_triplet_loss(targets.float(), embeddings.detach(), self.margin, self.squared)


def batch_hard_triplet_loss(labels, embeddings, margin, squared=False):
    """Build the triplet loss over a batch of embeddings.
    For each anchor, we get the hardest positive and hardest negative to form a triplet.
    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    # Get the pairwise distance matrix
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)

    device = labels.get_device()

    # For each anchor, get the hardest positive
    # First, we need to get a mask for every valid positive (they should have same label)
    mask_anchor_positive = _get_anchor_positive_triplet_mask(labels, device).float()

    # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
    anchor_positive_dist = mask_anchor_positive * pairwise_dist

    # shape (batch_size, 1)
    hardest_positive_dist, _ = anchor_positive_dist.max(1, keepdim=True)

    # For each anchor, get the hardest negative
    # First, we need to get a mask for every valid negative (they should have different labels)
    mask_anchor_negative = _get_anchor_negative_triplet_mask(labels).float()

    # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
    max_anchor_negative_dist, _ = pairwise_dist.max(1, keepdim=True)
    anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

    # shape (batch_size,)
    hardest_negative_dist, _ = anchor_negative_dist.min(1, keepdim=True)

    # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
    tl = hardest_positive_dist - hardest_negative_dist + margin
    tl[tl < 0] = 0
    triplet_loss = tl.mean()

    return triplet_loss


def batch_all_triplet_loss(labels, embeddings, margin, squared=False):
    """Build the triplet loss over a batch of embeddings.
    We generate all the valid triplets and average the loss over the positive ones.
    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    # Get the pairwise distance matrix
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)

    anchor_positive_dist = pairwise_dist.unsqueeze(2)
    anchor_negative_dist = pairwise_dist.unsqueeze(1)

    # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
    # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
    # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
    # and the 2nd (batch_size, 1, batch_size)
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    # Put to zero the invalid triplets
    # (where label(a) != label(p) or label(n) == label(a) or a == p)
    mask = _get_triplet_mask(labels)
    triplet_loss = mask.float() * triplet_loss

    # Remove negative losses (i.e. the easy triplets)
    triplet_loss[triplet_loss < 0] = 0

    # Count number of positive triplets (where triplet_loss > 0)
    valid_triplets = triplet_loss[triplet_loss > 1e-16]
    num_positive_triplets = valid_triplets.size(0)
    num_valid_triplets = mask.sum()

    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets.float() + 1e-16)

    # Get final mean triplet loss over the positive valid triplets
    triplet_loss = triplet_loss.sum() / (num_positive_triplets + 1e-16)

    return triplet_loss, fraction_positive_triplets


def _pairwise_distances(embeddings, squared=False):
    """Compute the 2D matrix of distances between all the embeddings.
    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    dot_product = torch.matmul(embeddings, embeddings.t())

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (batch_size,)
    square_norm = torch.diag(dot_product)

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances[distances < 0] = 0

    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = distances.eq(0).float()
        distances = distances + mask * 1e-16

        distances = (1.0 -mask) * torch.sqrt(distances)

    return distances


def _get_triplet_mask(labels):
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    """
    # Check that i, j and k are distinct
    indices_equal = torch.eye(labels.size(0)).byte()
    indices_not_equal = ~indices_equal
    i_not_equal_j = indices_not_equal.unsqueeze(2)
    i_not_equal_k = indices_not_equal.unsqueeze(1)
    j_not_equal_k = indices_not_equal.unsqueeze(0)

    distinct_indices = (i_not_equal_j & i_not_equal_k) & j_not_equal_k

    label_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
    i_equal_j = label_equal.unsqueeze(2)
    i_equal_k = label_equal.unsqueeze(1)

    valid_labels = ~i_equal_k & i_equal_j

    return valid_labels & distinct_indices


def _get_anchor_positive_triplet_mask(labels, device):
    """Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check that i and j are distinct
    indices_equal = torch.eye(labels.size(0)).bool().to(device)
    indices_not_equal = ~indices_equal

    # Check if labels[i] == labels[j]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    #labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
    labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)

    return labels_equal & indices_not_equal


def _get_anchor_negative_triplet_mask(labels):
    """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check if labels[i] != labels[k]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)

    return ~(labels.unsqueeze(0) == labels.unsqueeze(1))