"""Implementation of the MultiConcat classifier."""
import torch
from torch import nn
from torch.nn import functional as nnf
import torch_geometric as pyg
from morphoclass import layers
from src.utils.multi_neutrite_attribute import MultiNeutriteAttribute
from typing import Literal

class LinearEmbedder(nn.Module):
    """The embedder part of the `MultiConcat` classifier.

    This is a simple linear layer with one hidden layer and ReLU activation
    that is used to transform the input data into a higher dimensional space.
    """

    def __init__(self, 
        vectorization,
        embedding_dim=512,
        hidden_dim=256,
        dropout=0.3, 
        OPT_neutrite: Literal["axon","apical","basal"] = None, ## for neutrite-type specific attributes of type MultiNeutriteAttribute
        ):
        super().__init__()
        self.OPT_neutrite = OPT_neutrite
        self.vectorization = vectorization
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.hidden = nn.LazyLinear(self.hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.output = nn.Linear(self.hidden_dim, embedding_dim)

    def forward(self, data):
        """Do the forward pass.

        Parameters
        ----------
        data : torch_geometric.data.Batch | torch.Tensor
            A batch of the MorphologyDataset dataset.

        Returns
        -------
        x : torch.Tensor
            The embedding of the persistence image. The dimension of
            the tensor is (n_batch,channels=1, 512).
        """
        
        if hasattr(data, self.vectorization):
            x = getattr(data, self.vectorization)
            ## VARIANT A: (not yet supported) 
            # if type(x) == MultiNeutriteAttribute:  ##for neutrite specific attributes
            #     x = getattr(x, self.neutriteOPT)
        elif hasattr(data, self.vectorization+"_"+ self.neutriteOPT):
                x= getattr(data, self.vectorization+"_"+ self.neutriteOPT)
        else: 
            print("Attributes of data:", dir(data))
            print(self.vectorization)
            raise ValueError(f"The input data does not have a {self.vectorization} attribute.")

        # Handle shape: (batch, 1, size) → (batch, size)
        if x.dim() == 3 and x.shape[1] == 1:
            x = x.squeeze(1)  # remove channel dimension for 1D vectorization
        else: 
            raise ValueError(f"Expected input shape (batch, 1, size), but got {x.shape}")
        
        
        x = self.hidden(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.output(x)
        return x

class CNNEmbedder(nn.Module):
    """The embedder part of the `CNNet` classifier."""

    def __init__(self,
            embedding_dim=512,
            OPT_neutrite: Literal["axon","apical","basal"]  = None ## for neutrite-type specific attributes of type MultiNeutriteAttribute
            ):
        super().__init__()
        self.OPT_neutrite = OPT_neutrite
        self.embedding_dim = embedding_dim
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.fc = nn.LazyLinear(embedding_dim) #in_features=3 * (image_size // 4) ** 2
        

    def forward(self, data):
        """Do the forward pass.

        Parameters
        ----------
        data : torch_geometric.data.Batch | torch.Tensor
            A batch of the MorphologyDataset dataset.

        Returns
        -------
        x : torch.Tensor
            The embedding of the persistence image. The dimension of
            the tensor is (n_batch, 512).
        """
        
        
        if hasattr(data, "persistence_image"):
            x = data.persistence_image
            ## VARIANT A: (not yet supported) 
            # if type(x) == MultiNeutriteAttribute:  ##for neutrite specific attributes
            #     x = getattr(x, self.neutriteOPT)
        elif hasattr(data, "persistence_image_"+ self.OPT_neutrite):
                x= getattr(data, "persistence_image_"+ self.OPT_neutrite)
        else: return ValueError("The input data does not have a persistence image.")

        x = self.conv1(x)
        x = nnf.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = nnf.relu(x)
        x = self.pool2(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)

        return x

class GraphEmbedder(nn.Module):
    """The embedder for the `ManNet` network.

    The embedder consists of two bidirectional ChebConv blocks followed
    by a global pooling layer.

    Parameters
    ----------
    n_features : int
        The number of input features.
    pool_name : {"avg", "sum", "att"}
        The type of pooling layer to use:
        - "avg": global average pooling
        - "sum": global sum pooling
        - "att": global attention pooling (trainable)
    lambda_max : float or list of float or None
        Originally the highest eigenvalue(s) of the adjacency matrix. In
        ChebConvs this value is usually computed from the adjacency matrix
        directly and used for normalization. This however doesn't work for
        non-symmetric matrices and we fix a constant value instead of computing
        it. Experiments show that there is no impact on performance.
    normalization : {None, "sym", "rw"}
        The normalization type of the graph Laplacian to use in the ChebConvs.
        Possible values:
        - None: no normalization
        - "sym": symmetric normalization
        - "rw": random walk normalization
    flow : {"target_to_source", "source_to_target"}
        The message passing flow direction in ChebConvs for directed graphs.
    edge_weight_idx : int or None
        The index of the edge feature tensor (`data.edge_attr`) to use as
        edge weights.
    """

    def __init__(
        self,
        embedding_dim=512,
        cheb_conv_hidden_dim=128,
        n_features=1,
        pool_name="avg",
        lambda_max=3.0,
        normalization="sym",
        flow="target_to_source",
        edge_weight_idx=None,
        OPT_neutrite: Literal["axon","apical","basal"]  = None ## for neutrite-type specific attributes of type MultiNeutriteAttribute
    ):
        super().__init__()
        self.OPT_neutrite = OPT_neutrite
        self.n_features = n_features

        self.pool_name = pool_name
        self.lambda_max = lambda_max
        self.normalization = normalization
        self.flow = flow
        self.edge_weight_idx = edge_weight_idx

        conv_kwargs = {
            "K": 5,
            "flow": self.flow,
            "normalization": self.normalization,
            "lambda_max": self.lambda_max,
        }

        self.bi1 = layers.BidirectionalBlock(n_features, cheb_conv_hidden_dim, **conv_kwargs)
        self.bi2 = layers.BidirectionalBlock(cheb_conv_hidden_dim, embedding_dim, **conv_kwargs)
        self.relu = nn.ReLU()

        if pool_name == "avg":
            self.pool = pyg.nn.global_mean_pool
        elif pool_name == "sum":
            self.pool = pyg.nn.global_add_pool
        elif pool_name == "att":
            self.pool = layers.AttentionGlobalPool(embedding_dim)
        else:
            raise ValueError(f"Unknown pooling method ({pool_name})")
    
    def forward(self, data):
        """Run the forward pass.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input batch of data.

        Returns
        -------
        torch.Tensor
            The computed graph embeddings of the input morphologies. The
            shape is (n_samples, 512).
        """
        # Note: case in else section can be removed as soon as they implement
        # the feature, see https://github.com/pytorch/captum/issues/494
        if hasattr(data, "x"):
            x = data.x
            edge_index = data.edge_index
            edge_attr = data.edge_attr
            # # VARIANT A: (not yet supported)
            # if type(x) == MultiNeutriteAttribute:  ##for neutrite specific attributes
            #     x = getattr(x, self.OPT_neutrite)
            #     assert type(edge_index) == MultiNeutriteAttribute
            #     edge_index = getattr(edge_index, self.OPT_neutrite)
            #     assert type(edge_attr) == MultiNeutriteAttribute
            #     edge_attr = getattr(edge_attr, self.OPT_neutrite)
            # # VARIANT B: 
            
        elif getattr(data, "x_" + self.OPT_neutrite):
            x = getattr(data, "x_" + self.OPT_neutrite)
            edge_index = getattr(data, "edge_index_" + self.OPT_neutrite)
            edge_attr = getattr(data, "edge_attr_" + self.OPT_neutrite)

        batch = data.batch
        edge_weight = None
        if edge_attr is not None and self.edge_weight_idx is not None:
             edge_weight = edge_attr[:, self.edge_weight_idx]

        x = self.bi1(x, edge_index, edge_weight=edge_weight)
        x = self.relu(x)
        x = self.bi2(x, edge_index, edge_weight=edge_weight)
        x = self.relu(x)
        x = self.pool(x, batch)
        return x
 


class MultiConcatBackbone(nn.Module):
    def __init__(self,
            embedding_dim=512,
            linear_hidden_dim=256,
            cheb_conv_hidden_dim=128,
            dropout=0.3,
            n_node_features=1, 
            pool_name="avg",
            lambda_max=3.0,
            normalization="sym",
            flow="target_to_source",
            embeddings: list[str] = ["persistence_image"],
            normalize_emb_weights: bool = True,
            normalize_emb_temp: float = 1.0,
            OPT_neutrite: Literal["axon","apical","basal"]  = None ## for neutrite-type specific attributes of type MultiNeutriteAttribute
            ):
        super().__init__()
        self.n_node_features = n_node_features
        self.embeddings = embeddings
        self.normalize_emb_weights = normalize_emb_weights
        self.normalize_emb_temp = normalize_emb_temp
        
        for emb in self.embeddings:
            if emb == "gnn":
                self.add_module(
                    emb + "_embedder", 
                    GraphEmbedder(embedding_dim,cheb_conv_hidden_dim,n_node_features,pool_name,lambda_max,normalization,flow,edge_weight_idx=0,OPT_neutrite=OPT_neutrite))
            elif emb == "persistence_image":
                self.add_module(emb + "_embedder", CNNEmbedder(embedding_dim,OPT_neutrite))
            else:
                self.add_module(emb + "_embedder", LinearEmbedder(emb,embedding_dim,linear_hidden_dim,dropout,OPT_neutrite))
            setattr(self, emb + "_weight", nn.Parameter(torch.randn(1) * 0.1+1, requires_grad=True))

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
        if len(self.embeddings) == 1:
            x = getattr(self, self.embeddings[0] + "_embedder")(data)
            return x
        raw_weights = torch.stack([getattr(self, emb + "_weight") for emb in self.embeddings]).squeeze()
        weights = nnf.softmax(raw_weights/self.normalize_emb_temp, dim=0) if self.normalize_emb_weights else raw_weights

        x_to_concat = []
        for i, emb in enumerate(self.embeddings):
            emb_out = getattr(self, emb + "_embedder")(data)
            x_to_concat.append(weights[i] * emb_out)
        x = torch.cat(x_to_concat, dim=1)
        return x
    
    def print_embedding_weights(self):
        weights = [getattr(self, emb + "_weight") for emb in self.embeddings]
        if self.normalize_emb_weights:
            norm_weights = weights
        else:
            norm_weights = nnf.softmax(torch.stack(weights).squeeze(), dim=0)

        print("\n=== Embedding Contribution Weights (Normalized) ===")
        for emb, nw, w in zip(self.embeddings, norm_weights,weights):
            if self.normalize_emb_weights:
                print(f"{emb}: {w.item():.4f} ({nw.item():.2f}/1.00)")
            else:
                print(f"{emb}: {w.item():.4f} ({nw.item():.2f}/1.00)")
        print("===\n")


class MultiConcat(nn.Module):
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
            cls_hidden_dim=512,
            linear_hidden_dim=256,
            cheb_conv_hidden_dim=128,
            ## MAN EMBEDDER
            n_node_features=1, 
            pool_name="avg",
            lambda_max=3.0,
            normalization="sym",
            flow="target_to_source",
            ## LINEAR EMBEDDER for which veqctorizations?
            embeddings: list[str] = ["gnn","persistence_image"],
            normalize_emb_weights: bool = True,
            normalize_emb_temp: float = 1.0
        ):
        super().__init__()
        self.bn = bn
        self.dropout = nn.Dropout(p=dropout)  
        self.embeddings = embeddings

        self.feature_extractor = MultiConcatBackbone(
            embedding_dim=embedding_dim,
            linear_hidden_dim=linear_hidden_dim,
            cheb_conv_hidden_dim=cheb_conv_hidden_dim,
            dropout=embedding_dropout,
            ## MAN EMBEDDER
            n_node_features=n_node_features, 
            pool_name=pool_name,
            lambda_max=lambda_max,
            normalization=normalization,
            flow=flow,
            ## lin EMBEDDER
            embeddings=embeddings,
            normalize_emb_weights=normalize_emb_weights,
            normalize_emb_temp=normalize_emb_temp
        )

        n_out_features = len(embeddings) * embedding_dim
        if self.bn:
            self.bn = nn.BatchNorm1d(num_features=n_out_features)
        self.hidden = nn.Linear(in_features=n_out_features, out_features=cls_hidden_dim)
        self.classify = nn.Linear(in_features=cls_hidden_dim, out_features=n_classes)

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
            if hasattr(self, "feature_extractor") and len(self.embeddings) > 1:
                self.feature_extractor.print_embedding_weights()
        except Exception as e:
            print(f"[Warning] Failed to print embedding weights on __del__: {e}")
            