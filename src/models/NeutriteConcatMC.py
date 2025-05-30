"""Implementation of the MultiConcat classifier."""
from src.models.MultiConcat import  MultiConcatBackbone
import torch
from torch import nn
from torch.nn import functional as nnf
from typing import Literal,List


class NeutriteConcatBackbone(nn.Module):
    def __init__(self,
            embedding_dim=512,
            dropout=0.3,
            n_node_features=1, 
            pool_name="avg",
            lambda_max=3.0,
            normalization="sym",
            flow="target_to_source",
            embeddings: list[str] = ["persistence_image"],
            normalize_emb_weights: bool = True,
            normalize_emb_temp: float = 0.1,
            ntype_list: List[Literal["axon","apical","basal"]]  =["axon","apical"] ## for neutrite-type specific attributes of type MultiNeutriteAttribute
            ):
        super().__init__()
        self.ntype_list = ntype_list
        self.normalize_emb_weights = normalize_emb_weights
        self.normalize_emb_temp = normalize_emb_temp
        

        for n_type in self.ntype_list:
            self.add_module(
                n_type + "_embedder",
                MultiConcatBackbone(
                    embedding_dim=embedding_dim,
                    dropout=dropout,
                    ## MAN EMBEDDER
                    n_node_features=n_node_features, 
                    pool_name=pool_name,
                    lambda_max=lambda_max,
                    normalization=normalization,
                    flow=flow,
                    ## lin EMBEDDER
                    embeddings=embeddings,
                    normalize_emb_weights=normalize_emb_weights,
                    OPT_neutrite=n_type
                )    
            )
            setattr(self, n_type + "_weight", nn.Parameter(torch.ones([1]), requires_grad=True))

    def forward(self, data):
        """Compute the forward pass.

        Parameters
        ----------
        data : torch_geometric.data.data.Data
            A batch of input graph data for the GNN layers.
        images
            A batch of input persistence images for the CNN layers.

        Returns
        -------
        log_softmax
            The log softmax of the predictions.
        """
        raw_weights = torch.stack([getattr(self, n_type + "_weight") for n_type in self.ntype_list]).squeeze()
        weights = nnf.softmax(raw_weights/self.normalize_emb_temp, dim=0) if self.normalize_emb_weights else raw_weights
        x_to_concat = []
        for i, n_type in enumerate(self.ntype_list):
            n_type_emb =  getattr(self, n_type + "_embedder")(data)
            x_to_concat.append(weights[i] * n_type_emb)
        x = torch.cat(x_to_concat, dim=1)
        return x
    
    def print_ntype_weights(self):
        weights = [getattr(self, n_type + "_weight") for n_type in self.ntype_list]
        norm_weights = nnf.softmax(torch.stack(weights).squeeze(), dim=0)
 
        print("\n=== Embedding Contribution Weights (Normalized) ===")
        for n_type, nw, w in zip(self.ntype_list, norm_weights,weights):
            if self.normalize_emb_weights:
                print(f"{n_type}: {w.item():.4f} ({nw.item():.2f}/1.00)")
            else:
                print(f"{n_type}: {w.item():.4f} ({nw.item():.2f}/1.00)")
        print("===\n")




class NeutriteConcatMC(nn.Module):
    """A neuron m-type classifier based on graph and image convolutions tdm-vectorizations and morphomatrics.

    In the feature extraction part of the network 
        > graph convolution layers are applied to the graph node features of the apical dendrites, 
        > the CNN layers are applied to the persistence image representation
        > the linear vectorization layers are applied to the other pd vectorizations 
        > linear vectorization layers are applied to the morphometrics features
    The resulting features are concatenated and passed
    through a fully-connected layer for classification.

    Parameters
    ----------
    n_node_features : int
        The number of input node features for the GNN layers.
    n_classes : int
        The number of output classes.
    image_size : int
        The width (or height) of the input persistence images. It is assumed
        that the images are square so that the width and height are equal.
    lin_vectorizations : list of str
        The list of vectorizations to be used for the linear vectorization (ATTRIBTUE NAMES)
    bn : bool, default False
        Whether or not to include a batch normalization layer between the
        feature extractor and the fully-connected classification layer.
    """
    def __init__(self,
            ## General 
            n_classes=4,
            bn=False,
            dropout=0.4,
            embedding_dropout=0.2,
            embedding_dim=512,
            ## MAN EMBEDDER
            n_node_features=1, 
            pool_name="avg",
            lambda_max=3.0,
            normalization="sym",
            flow="target_to_source",
            ## LINEAR EMBEDDER for which veqctorizations?
            embeddings: list[str] = ["gnn","persistence_image"],
            normalize_emb_weights: bool = True,
            normalize_emb_temp: float = 0.1,
            ntype_list: List[Literal["axon","apical", "basal"]]= ["axon","apical"]
        ):
        super().__init__()
        self.bn = bn
        self.dropout = nn.Dropout(p=dropout)  

        self.feature_extractor =  NeutriteConcatBackbone(
            embedding_dim=embedding_dim,
            dropout= embedding_dropout,
            n_node_features=n_node_features,
            pool_name=pool_name,
            lambda_max=lambda_max,
            normalization=normalization,
            flow=flow,
            embeddings= embeddings,
            normalize_emb_weights = normalize_emb_weights,
            normalize_emb_temp = normalize_emb_temp,
            ntype_list= ntype_list ## for neutrite-type specific attributes of type MultiNeutriteAttribute
            )

        nout_features=   len(ntype_list)* len(embeddings) * embedding_dim

        if self.bn:
            self.bn = nn.BatchNorm1d(num_features=nout_features)
        self.hidden = nn.Linear(in_features=nout_features, out_features=embedding_dim)
        self.classify = nn.Linear(in_features=embedding_dim, out_features=n_classes)

    def forward(self, data):
        """Compute the forward pass.

        Parameters
        ----------
        data : torch_geometric.data.data.Data
            A batch of input graph data for the GNN layers.
        images
            A batch of input persistence images for the CNN layers.

        Returns
        -------
        log_softmax
            The log softmax of the predictions.
        """
        x = self.feature_extractor(data)

        if self.bn and x.shape[0] > 1:
            x = self.bn(x)
            x = nnf.relu(x)
        x = self.hidden(x)
        x = nnf.relu(x)
        x = self.dropout(x)   
        x = self.classify(x)

        return nnf.log_softmax(x, dim=1)
    
    def __del__(self):
        try:
            if hasattr(self, "feature_extractor"):
                self.feature_extractor.print_ntype_weights()
        except Exception as e:
            print(f"[Warning] Failed to print embedding weights on __del__: {e}")
            