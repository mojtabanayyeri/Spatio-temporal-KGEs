"""Euclidean Knowledge Graph embedding models where embeddings are in complex space."""
import torch
from torch import nn

from models.base import KGModel

QUAD_MODELS = ["QuatE", "DES"]


class BaseQ(KGModel):
    """Hypercomplex Knowledge Graph Embedding models.

    Attributes:
        embeddings: hypercomplex embeddings for entities and relations
    """

    def __init__(self, args):
        """Initialize a hypercomplex KGModel."""
        super(BaseQ, self).__init__(args.sizes, args.rank, args.dropout, args.gamma, args.dtype, args.bias,
                                    args.init_size)
        assert self.rank % 4 == 0, "Hypercomplex models require 4D embedding dimension"
        self.rank = self.rank // 4
        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 4 * self.rank, sparse=True)
            for s in self.sizes[:2]
        ])
        self.embeddings[0].weight.data = self.init_size * self.embeddings[0].weight.to(self.data_type)
        self.embeddings[1].weight.data = self.init_size * self.embeddings[1].weight.to(self.data_type)

    def get_rhs(self, queries, eval_mode):
        """Get embeddings and biases of target entities."""
        if eval_mode:
            return self.embeddings[0].weight, self.bt.weight
        else:
            return self.embeddings[0](queries[:, 2]), self.bt(queries[:, 2])

    def similarity_score(self, lhs_e, rhs_e, eval_mode):
        """Compute similarity scores or queries against targets in embedding space."""
        lhs_e = lhs_e[:, :self.rank], lhs_e[:, self.rank:2*self.rank], lhs_e[:, 2*self.rank:3*self.rank], lhs_e[:, 3*self.rank:]
        rhs_e = rhs_e[:, :self.rank], rhs_e[:, self.rank:2*self.rank], rhs_e[:, 2*self.rank:3*self.rank], rhs_e[:, 3*self.rank:]
        if eval_mode:
            return lhs_e[0] @ rhs_e[0].transpose(0, 1) + lhs_e[1] @ rhs_e[1].transpose(0, 1) + lhs_e[2] @ rhs_e[2].transpose(0, 1) + lhs_e[3] @ rhs_e[3].transpose(0, 1)
        else:
            return torch.sum(
                lhs_e[0] * rhs_e[0] + lhs_e[1] * rhs_e[1] + lhs_e[2] * rhs_e[2] + lhs_e[3] * rhs_e[3],
                1, keepdim=True
            )

    def get_quad_embeddings(self, queries):        
        """Get quad embeddings of queries."""        
        head_e = self.embeddings[0](queries[:, 0])        
        rel_e = self.embeddings[1](queries[:, 1])        
        rhs_e = self.embeddings[0](queries[:, 2])        
        head_e = head_e[:, :self.rank//2], head_e[:, self.rank//2:self.rank], head_e[:, self.rank:3*self.rank//2], head_e[:, 3*self.rank//2:]
        rel_e = rel_e[:, :self.rank//2], rel_e[:, self.rank//2:self.rank], rel_e[:, self.rank:3*self.rank//2], rel_e[:, 3*self.rank//2:]
        rhs_e = rhs_e[:, :self.rank//2], rhs_e[:, self.rank//2:self.rank], rhs_e[:, self.rank:3*self.rank//2], rhs_e[:, 3*self.rank//2:]   
        return head_e, rel_e, rhs_e


    def get_factors(self, queries):
        """Compute factors for embeddings' regularization."""
        head_e, rel_e, rhs_e = self.get_quad_embeddings(queries)
        head_f = torch.sqrt(head_e[0] ** 2 + head_e[1] ** 2 + head_e[2] ** 2 + head_e[3] ** 2)
        rel_f = torch.sqrt(rel_e[0] ** 2 + rel_e[1] ** 2 + rel_e[2] ** 2 + rel_e[3] ** 2)
        rhs_f = torch.sqrt(rhs_e[0] ** 2 + rhs_e[1] ** 2 + rhs_e[2] ** 2 + rhs_e[3] ** 2)
        return head_f, rel_f, rhs_f

class DES(BaseQ):
    """Simple complex model http://proceedings.mlr.press/v48/trouillon16.pdf"""
    def get_queries(self, queries):
        """Compute embedding and biases of queries."""
        head_e, rel_e, _ = self.get_quad_embeddings(queries)
                
        s_a = head_e[0]        
        x_a = head_e[1]        
        y_a = head_e[2]        
        z_a = head_e[3]

        
        s_b = rel_e[0]        
        x_b = rel_e[1]        
        y_b = rel_e[2]        
        z_b = rel_e[3]

        A = s_a * s_b - x_a * x_b + y_a * y_b + z_a * z_b
        B = s_a * x_b + s_b * x_a - y_a * z_b + y_b * z_a
        C = s_a * y_b + s_b * y_a + z_a * x_b - z_b * x_a
        D = s_a * z_b + s_b * z_a + x_a * y_b - x_b * y_a


        lhs_e = torch.cat([
            A,
            B,
            -C,
            -D
            ], 1)    
        return lhs_e, self.bh(queries[:, 0])


class RotatE(BaseQ):
    """Rotations in complex space https://openreview.net/pdf?id=HkgEQnRqYQ"""

    def get_queries(self, queries):
        """Compute embedding and biases of queries."""
        head_e, rel_e, _ = self.get_complex_embeddings(queries)
        rel_norm = torch.sqrt(rel_e[0] ** 2 + rel_e[1] ** 2)
        cos = rel_e[0] / rel_norm
        sin = rel_e[1] / rel_norm
        lhs_e = torch.cat([
            head_e[0] * cos - head_e[1] * sin,
            head_e[0] * sin + head_e[1] * cos
        ], 1)
        return lhs_e, self.bh(queries[:, 0])
