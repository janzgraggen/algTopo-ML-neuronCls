"""Implementation of the MultiConcat classifier."""
import torch
from torch import nn
from torch.nn import functional as nnf
import torch_geometric as pyg
from morphoclass import layers

class Linear1DVectorizationEmbedder(nn.Module):
    """The embedder part of the `MultiConcat` classifier.

    This is a simple linear layer with one hidden layer and ReLU activation
    that is used to transform the input data into a higher dimensional space.
    """

    def __init__(self, vectorization,embedding_dim=512):
        super().__init__()
        self.vectorization = vectorization
        self.embedding_dim = embedding_dim
        self.hiddeen_dim = embedding_dim // 2
        #self.hidden = nn.Linear(n_features, 256)
        self.relu = nn.ReLU()
        self.output = nn.Linear(self.hiddeen_dim, embedding_dim)

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
            n_features = x.shape[-1]
        else: 
            print("Attributes of data:", dir(data))
            print(self.vectorization)
            raise ValueError(f"The input data does not have a {self.vectorization} attribute.")

        # Handle shape: (batch, 1, size) → (batch, size)
        if x.dim() == 3 and x.shape[1] == 1:
            x = x.squeeze(1)  # remove channel dimension for 1D vectorization
        else: 
            raise ValueError(f"Expected input shape (batch, 1, size), but got {x.shape}")
        
        self.hidden = nn.Linear(n_features, self.hiddeen_dim)

        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        return x

class CNNEmbedder(nn.Module):
    """The embedder part of the `CNNet` classifier."""

    def __init__(self,embedding_dim=512):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        #self.fc = nn.Linear(in_features=3 * (image_size // 4) ** 2, out_features=512)
        

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
        else: return ValueError("The input data does not have a persistence image.")

        image_size = x.shape[2]
        self.fc = nn.Linear(in_features=3 * (image_size // 4) ** 2, out_features=self.embedding_dim)

        x = self.conv1(x)
        x = nnf.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = nnf.relu(x)
        x = self.pool2(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)

        return x

class ManEmbedder(nn.Module):
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
        n_features=1,
        pool_name="avg",
        lambda_max=3.0,
        normalization="sym",
        flow="target_to_source",
        edge_weight_idx=None,
    ):
        super().__init__()

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

        self.bi1 = layers.BidirectionalBlock(n_features, 128, **conv_kwargs)
        self.bi2 = layers.BidirectionalBlock(128, embedding_dim, **conv_kwargs)
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
            ## MAN EMBEDDER     
            n_node_features = 1, 
            pool_name="avg",
            lambda_max=3.0,
            normalization="sym",
            flow="target_to_source",
            edge_weight_idx=None,
        
            ## LINEAR EMBEDDER for which veqctorizations? 
            vectorizations: list[str] = ["persistence_image"],
            ):
        
        super().__init__()
        self.n_node_features = n_node_features
        self.vectorizations = vectorizations

        self.gnn_embedder = ManEmbedder(
            embedding_dim=embedding_dim,
            n_features=n_node_features,
            pool_name=pool_name,
            lambda_max=lambda_max,
            normalization=normalization,
            flow=flow,
            edge_weight_idx=edge_weight_idx,)
        
        for v in self.vectorizations:
            if v == "persistence_image":
                self.add_module(
                    name=   v + "_embedder",
                    module= CNNEmbedder(embedding_dim),
                )  
            else:
                self.add_module(
                    name=   v + "_embedder",
                    module= Linear1DVectorizationEmbedder(vectorization=v),
                )

        self.gnn_weight = nn.Parameter(torch.ones([1]), requires_grad=True)
        #self.weight_cnn = nn.Parameter(torch.ones([1]), requires_grad=True) -> is in vectorization: persistence_image
        for v in self.vectorizations:
            setattr(
                self,
                v + "_weight",
                nn.Parameter(torch.ones([1]), requires_grad=True),
            )
        


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
        #x_gnn
        x = self.gnn_weight * self.gnn_embedder(data)
        for vectorization in self.vectorizations:
            x_vectorization = getattr(self, vectorization + "_embedder")(data)
            x_vectorization = getattr(self, vectorization + "_weight") * x_vectorization
            x= torch.cat([x, x_vectorization], dim=1)
        return x

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
            n_classes = 4,
            bn=False,
            embedding_dim=512,
            ## MAN EMBEDDER     
            n_node_features = 1, 
            pool_name="avg",
            lambda_max=3.0,
            normalization="sym",
            flow="target_to_source",
            edge_weight_idx=None,
            ## LINEAR EMBEDDER for which veqctorizations? 
            vectorizations: list[str] = ["persistence_image"]
        ):
            super().__init__()
            self.bn = bn
            
            self.feature_extractor = MultiConcatBackbone(
                embedding_dim=embedding_dim,
                ## MAN EMBEDDER     
                n_node_features = n_node_features, 
                pool_name=pool_name,
                lambda_max=lambda_max,
                normalization=normalization,
                flow=flow,
                edge_weight_idx=edge_weight_idx,
                ## lin EMBEDDER
                vectorizations= vectorizations
                )
            
            n_out_features_layer = embedding_dim
            n_out_features = (1+len(vectorizations))  * n_out_features_layer
            if self.bn:
                self.bn = nn.BatchNorm1d(num_features=n_out_features)
            self.hidden = nn.Linear(in_features=n_out_features, out_features=n_out_features_layer)
            self.classify = nn.Linear(in_features=n_out_features_layer, out_features=n_classes)
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

            if self.bn and x.shape[0]>1:
                x = self.bn(x)
                x = nnf.relu(x)
            x = self.hidden(x)
            x = nnf.relu(x)
            x = self.classify(x)

            return nnf.log_softmax(x, dim=1)

            




# class MultiConcat(nn.Module):
#     """A neuron m-type classifier based on graph and image convolutions tdm-vectorizations and morphomatrics.

#     In the feature extraction part of the network 
#         > graph convolution layers are applied to the graph node features of the apical dendrites, 
#         > the CNN layers are applied to the persistence image representation
#         > the linear vectorization layers are applied to the other pd vectorizations 
#         > linear vectorization layers are applied to the morphometrics features
#     The resulting features are concatenated and passed
#     through a fully-connected layer for classification.

#     Parameters
#     ----------
#     n_node_features : int
#         The number of input node features for the GNN layers.
#     n_classes : int
#         The number of output classes.
#     image_size : int
#         The width (or height) of the input persistence images. It is assumed
#         that the images are square so that the width and height are equal.
#     lin_vectorizations : list of str
#         The list of vectorizations to be used for the linear vectorization (ATTRIBTUE NAMES)
#     bn : bool, default False
#         Whether or not to include a batch normalization layer between the
#         feature extractor and the fully-connected classification layer.
#     """

#     def __init__(self,
#             ## General 
#             n_classes = 4,
#             bn=False,
#             ## MAN EMBEDDER     
#             n_node_features = 1, 
#             pool_name="avg",
#             lambda_max=3.0,
#             normalization="sym",
#             flow="target_to_source",
#             edge_weight_idx=None,
#             ## CNN EMBEDDER
#             image_size = 0, 

#             ## LINEAR EMBEDDER for which veqctorizations? 
#             vectorizations: list[str] = ["persistence_image"],
#             ):
        
#         super().__init__()
#         self.n_node_features = n_node_features
#         self.n_classes = n_classes
#         self.image_size = image_size
#         self.vectorizations = vectorizations
#         self.bn = bn

#         self.gnn_embedder = ManEmbedder(
#             n_features=n_node_features,
#             pool_name=pool_name,
#             lambda_max=lambda_max,
#             normalization=normalization,
#             flow=flow,
#             edge_weight_idx=edge_weight_idx,)
        
#         for v in self.vectorizations:
#             if v == "persistence_image":
#                 self.add_module(
#                     name=   v + "_embedder",
#                     module= CNNEmbedder(),
#                 )  
#             else:
#                 self.add_module(
#                     name=   v + "_embedder",
#                     module= Linear1DVectorizationEmbedder(vectorization=v),
#                 )

#         self.weight_gnn = nn.Parameter(torch.ones([1]), requires_grad=True)
#         #self.weight_cnn = nn.Parameter(torch.ones([1]), requires_grad=True) -> is in vectorization: persistence_image
#         for v in self.vectorizations:
#             setattr(
#                 self,
#                 v + "_weight",
#                 nn.Parameter(torch.ones([1]), requires_grad=True),
#             )
        
#         n_out_features_layer = 512
#         n_out_features = (1+len(self.vectorizations))  * n_out_features_layer
#         if self.bn:
#             self.bn = nn.BatchNorm1d(num_features=n_out_features)
#         self.hidden = nn.Linear(in_features=n_out_features, out_features=n_out_features_layer)
#         self.classify = nn.Linear(in_features=n_out_features_layer, out_features=self.n_classes)

#     def forward(self, data):
#         """Compute the forward pass.

#         Parameters
#         ----------
#         data : torch_geometric.data.data.Data
#             A batch of input graph data for the GNN layers.
#         images
#             A batch of input persistence images for the CNN layers.

#         Returns
#         -------
#         log_softmax
#             The log softmax of the predictions.
#         """
#         #x_gnn
#         x = self.weight_gnn * self.gnn_embedder(data)
#         for vectorization in self.vectorizations:
#             x_vectorization = getattr(self, vectorization + "_embedder")(data)
#             x_vectorization = getattr(self, vectorization + "_weight") * x_vectorization
#             x= torch.cat([x, x_vectorization], dim=1)

#         self.feature_extractor = x

#         if self.bn:
#             x = self.bn(x)
#             x = nnf.relu(x)
#         x = self.hidden(x)
#         x = nnf.relu(x)
#         x = self.classify(x)

#         return nnf.log_softmax(x, dim=1)
