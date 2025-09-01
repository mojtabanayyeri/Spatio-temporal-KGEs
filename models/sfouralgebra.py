"""Euclidean Knowledge Graph embedding models where embeddings are in complex space."""
import torch
from torch import nn
import numpy as np

from models.base import KGModel
from utils.euclidean import givens_DE_product

SFOURE_MODELS = ["sFourDE", "sFourDELoc", "sFourDETim"]


class BaseC(KGModel):
    """Complex Knowledge Graph Embedding models.

    Attributes:
        embeddings: complex embeddings for entities and relations
    """

    def __init__(self, args):
        """Initialize a Complex KGModel."""
        super(BaseC, self).__init__(args.sizes, args.rank, args.dropout, args.gamma, args.dtype, args.bias,
                                    args.init_size, args)
        assert self.rank % 4 == 0, "Complex models require even embedding dimension"
        #self.sizes = list(set(self.sizes))
        self.model_name = args.model              
        self.sizes = np.array(list(self.sizes))#np.expand_dims(np.array(list(self.sizes)), axis = 1)
        self.sizes = list(np.delete(self.sizes, 2, 0))
        self.rank = self.rank//4
        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 4*self.rank, sparse=True)
            for s in self.sizes
        ])
        
        self.embeddings[0].weight.data = self.init_size * self.embeddings[0].weight.to(self.data_type)
        self.embeddings[1].weight.data = self.init_size * self.embeddings[1].weight.to(self.data_type)              


        if len(self.sizes) == 4:          
            self.embeddings[2].weight.data = self.init_size * self.embeddings[2].weight.to(self.data_type)
            self.embeddings[3].weight.data = self.init_size * self.embeddings[3].weight.to(self.data_type)

    def get_rhs(self, queries, eval_mode):
        """Get embeddings and biases of target entities."""
        if self.model_name == 'sFourDELoc':
            if eval_mode:
                return self.embeddings[2].weight, self.bt.weight
            else:
                return self.embeddings[2](queries[:, 3]), self.bt(queries[:, 3])
        elif self.model_name == 'sFourDETim':                                        
              if eval_mode:                                
                  return self.embeddings[3].weight, self.bt.weight                                
              else:
                  #print(self.bt(queries[:, 4]))
                  #exit()
                  return self.embeddings[3](queries[:, 4]), self.bt(queries[:, 4])             
        else:
            if eval_mode:
                return self.embeddings[0].weight, self.bt.weight
            else:
                return self.embeddings[0](queries[:, 2]), self.bt(queries[:, 2])


    def similarity_score(self, lhs_e, rhs_e, eval_mode):
        """Compute similarity scores or queries against targets in embedding space."""                
        rhs_e = rhs_e[:, :self.rank], rhs_e[:, self.rank:2*self.rank], rhs_e[:, 2*self.rank:3*self.rank], rhs_e[:, 3*self.rank:]
        A,B,C,D = givens_DE_product(rhs_e, lhs_e, eval_mode)
        if eval_mode:            
            return A + B + C + D 
        else:
            return torch.sum(
                A + B + C + D,
                1, keepdim=True
            )

    def get_factors(self, queries):
        head_e = self.entity(queries[:, 0])
        rel_e = self.rel(queries[:, 1])
        rhs_e = self.entity(queries[:, 2])
        loc_e = self.loc_emb(queries[:, 3])
        tim_e = self.tim_emb(queries[:, 4])
        return head_e, rel_e, rhs_e, loc_e, tim_e

class sFourDE(BaseC):
    """Euclidean translations https://www.utc.fr/~bordesan/dokuwiki/_media/en/transe_nips13.pdf"""

    def __init__(self, args):
        super(sFourDE, self).__init__(args)
                      
    def get_queries(self, queries):
        head_e = self.entity(queries[:, 0])
        rel_e = self.rel(queries[:, 1])

        #Added
        loc_e = self.loc_emb(queries[:, 3])
        tim_e = self.tim_emb(queries[:, 4])
        #Ended       
        rank = self.rank
        head_e_s = head_e[:, :rank]

        rel_e_s = rel_e[:, :rank]
        loc_e_s = loc_e[:, :rank]
        tim_e_s = tim_e[:, :rank]
        
        lhs_e = head_e_s, rel_e_s, loc_e_s, tim_e_s
        lhs_biases = self.bh(queries[:, 0])
        return lhs_e, lhs_biases

    def save_embedding(self):        
         torch.save(self.entity, 'entity.pkl')         
         torch.save(self.rel, 'relation.pkl')         
         torch.save(self.loc_emb, 'location.pkl')         
         torch.save(self.tim_emb, 'time.pkl') 


class sFourDELoc(BaseC):
    """Euclidean translations https://www.utc.fr/~bordesan/dokuwiki/_media/en/transe_nips13.pdf"""

    def __init__(self, args):
        super(sFourDELoc, self).__init__(args)

    def get_queries(self, queries):
        head_e = self.entity(queries[:, 0])
        tail_e = self.entity(queries[:, 2])
        rel_e = self.rel(queries[:, 1])
        #Added
        tim_e = self.tim_emb(queries[:, 4])
        #Ended       
        rank = self.rank
        head_e_s = head_e[:, :rank]#, head_e[:, rank//2:rank], head_e[:, rank:3*rank//2], head_e[:, 3*rank//2:]
        rel_e_s = rel_e[:, :rank]#, loc_rot_e[:, rank//2:rank], tim_rot_e[:, :rank//2], tim_rot_e[:, rank//2:rank]
        tail_e_s = tail_e[:, :rank]
        tim_e_s = tim_e[:, :rank]

        lhs_e = head_e_s, rel_e_s, tail_e_s, tim_e_s
        lhs_biases = self.bh(queries[:, 3])
        return lhs_e, lhs_biases

class sFourDETim(BaseC):
    """Euclidean translations https://www.utc.fr/~bordesan/dokuwiki/_media/en/transe_nips13.pdf"""

    def __init__(self, args):
        super(sFourDETim, self).__init__(args)

        #Ended

    def get_queries(self, queries):
        head_e = self.entity(queries[:, 0])
        tail_e = self.entity(queries[:, 2])
        rel_e = self.rel(queries[:, 1])

        #Added
        loc_e = self.loc_emb(queries[:, 3])
        #Ended       
        rank = self.rank
        head_e_s = head_e[:, :rank]
        rel_e_s = rel_e[:, :rank]
        tail_e_s = tail_e[:, :rank]
        loc_e_s = loc_e[:, :rank]

        lhs_e = head_e_s, rel_e_s, tail_e_s, loc_e_s

        lhs_biases = self.bh(queries[:, 4])
        return lhs_e, lhs_biases

