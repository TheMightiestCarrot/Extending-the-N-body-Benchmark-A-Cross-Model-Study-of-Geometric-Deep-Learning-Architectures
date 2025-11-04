import numpy as np
import torch
import torch.nn as nn
from e3nn.nn import BatchNorm
from e3nn.o3 import Irreps
from torch_geometric.nn import MessagePassing, global_add_pool, global_mean_pool

from models.balanced_irreps import WeightBalancedIrreps

from .instance_norm import InstanceNorm
from .o3_building_blocks import O3TensorProduct, O3TensorProductSwishGate


class SEGNN(nn.Module):
    """Steerable E(3) equivariant message passing network"""

    def __init__(
        self,
        input_irreps=Irreps("2x1o + 1x0e"),
        hidden_features=64,
        lmax_h=1,
        lmax_attr=1,
        num_layers=4,
        output_irreps=Irreps("2x1o"),
        norm="batch",
        pool="avg",
        task="node",
        additional_message_irreps=Irreps("2x0e"),
        training_args=None,
    ):
        super().__init__()
        self.hidden_features = hidden_features
        self.lmax_h = lmax_h
        self.lmax_attr = lmax_attr
        self.num_layers = num_layers

        self.node_attr_irreps = Irreps.spherical_harmonics(lmax_attr)

        hidden_irreps = WeightBalancedIrreps(
            Irreps("{}x0e".format(hidden_features)),
            self.node_attr_irreps,
            sh=True,
            lmax=lmax_h,
        )
        edge_attr_irreps = Irreps.spherical_harmonics(lmax_attr)
        self.task = task
        self.hidden_irreps = hidden_irreps
        self.edge_attr_irreps = edge_attr_irreps
        self.node_attr_irreps = self.node_attr_irreps
        self.norm = norm
        self.additional_message_irreps = additional_message_irreps
        self.pool = pool
        self.training_args = training_args

        # Create network, embedding first
        # self.embedding_layer_1 = O3TensorProductSwishGate(
        #     input_irreps, hidden_irreps, node_attr_irreps
        # )
        # self.embedding_layer_2 = O3TensorProduct(
        #     hidden_irreps, hidden_irreps, node_attr_irreps
        # )

        self.embedding_layer = O3TensorProduct(
            input_irreps, hidden_irreps, self.node_attr_irreps
        )

        # Message passing layers.
        layers = []
        for _ in range(num_layers):
            layers.append(
                SEGNNLayer(
                    hidden_irreps,
                    hidden_irreps,
                    hidden_irreps,
                    edge_attr_irreps,
                    self.node_attr_irreps,
                    norm=norm,
                    additional_message_irreps=additional_message_irreps,
                )
            )
        self.layers = nn.ModuleList(layers)

        # Prepare for output irreps, since the attrs will disappear after pooling
        if task == "graph":
            pooled_irreps = (
                (output_irreps * hidden_irreps.num_irreps).simplify().sort().irreps
            )
            self.pre_pool1 = O3TensorProductSwishGate(
                hidden_irreps, hidden_irreps, self.node_attr_irreps
            )
            self.pre_pool2 = O3TensorProduct(
                hidden_irreps, pooled_irreps, self.node_attr_irreps
            )
            self.post_pool1 = O3TensorProductSwishGate(pooled_irreps, pooled_irreps)
            self.post_pool2 = O3TensorProduct(pooled_irreps, output_irreps)
            self.init_pooler(pool)
        elif task == "node":
            # first non-linear node update layer
            self.pre_pool1 = O3TensorProductSwishGate(
                hidden_irreps, hidden_irreps, self.node_attr_irreps
            )
            # second linear node update layer (this basically returns node update)
            self.pre_pool2 = O3TensorProduct(
                hidden_irreps, output_irreps, self.node_attr_irreps
            )

    def get_model_size(self):
        return self.hidden_features

    def get_serializable_attributes(self):
        return {
            "hidden_features": self.hidden_features,
            "lmax_h": self.lmax_h,
            "lmax_attr": self.lmax_attr,
            "node_attr_irreps": str(self.node_attr_irreps),
            "num_layers": self.num_layers,
            "input_irreps": str(self.embedding_layer.irreps_in1),
            "hidden_irreps": str(self.hidden_irreps),
            "output_irreps": str(self.pre_pool2.irreps_out),
            "edge_attr_irreps": str(self.edge_attr_irreps),
            "norm": self.norm if hasattr(self, "norm") else None,
            "pool": self.pooler.__name__ if hasattr(self, "pooler") else None,
            "task": self.task,
            "additional_message_irreps": str(self.additional_message_irreps),
            "training_args": self.training_args,
            "num_params": sum(p.numel() for p in self.parameters()),
        }

    def init_pooler(self, pool):
        """Initialise pooling mechanism"""
        if pool == "avg":
            self.pooler = global_mean_pool
        elif pool == "sum":
            self.pooler = global_add_pool

    def catch_isolated_nodes(self, graph):
        """Isolated nodes should also obtain attributes"""
        if (
            graph.has_isolated_nodes()
            and graph.edge_index.max().item() + 1 != graph.num_nodes
        ):
            nr_add_attr = graph.num_nodes - (graph.edge_index.max().item() + 1)
            add_attr = graph.node_attr.new_tensor(
                np.zeros((nr_add_attr, graph.node_attr.shape[-1]))
            )
            graph.node_attr = torch.cat((graph.node_attr, add_attr), -2)
        # Trivial irrep value should always be 1 (is automatically so for connected nodes, but isolated nodes are now 0)
        graph.node_attr[:, 0] = 1.0

    def forward(self, graph):
        """SEGNN forward pass"""
        x, pos, edge_index, edge_attr, node_attr, batch = (
            graph.x,
            graph.pos,
            graph.edge_index,
            graph.edge_attr,
            graph.node_attr,
            graph.batch,
        )
        try:
            additional_message_features = graph.additional_message_features
        except AttributeError:
            additional_message_features = None

        self.catch_isolated_nodes(graph)

        # Embed
        # x = self.embedding_layer_1(x, node_attr)
        # x = self.embedding_layer_2(x, node_attr)
        x = self.embedding_layer(x, node_attr)

        # Pass messages
        for layer in self.layers:
            x = layer(
                x, edge_index, edge_attr, node_attr, batch, additional_message_features
            )

        # Pre pool
        x = self.pre_pool1(x, node_attr)
        x = self.pre_pool2(x, node_attr)

        if self.task == "graph":
            # Pool over nodes
            x = self.pooler(x, batch)

            # Predict
            x = self.post_pool1(x)
            x = self.post_pool2(x)
        return x


class SEGNNLayer(MessagePassing):
    """E(3) equivariant message passing layer."""

    def __init__(
        self,
        input_irreps,
        hidden_irreps,
        output_irreps,
        edge_attr_irreps,
        node_attr_irreps,
        norm=None,
        additional_message_irreps=None,
    ):
        super().__init__(node_dim=-2, aggr="add")
        self.hidden_irreps = hidden_irreps

        # 2x because this tensor product calculates relationship between two neighbouring nodes
        message_input_irreps = (2 * input_irreps + additional_message_irreps).simplify()
        update_input_irreps = (input_irreps + hidden_irreps).simplify()

        self.message_layer_1 = O3TensorProductSwishGate(
            message_input_irreps, hidden_irreps, edge_attr_irreps
        )
        self.message_layer_2 = O3TensorProductSwishGate(
            hidden_irreps, hidden_irreps, edge_attr_irreps
        )
        self.update_layer_1 = O3TensorProductSwishGate(
            update_input_irreps, hidden_irreps, node_attr_irreps
        )
        self.update_layer_2 = O3TensorProduct(
            hidden_irreps, hidden_irreps, node_attr_irreps
        )

        self.setup_normalisation(norm)

    def setup_normalisation(self, norm):
        """Set up normalisation, either batch or instance norm"""
        self.norm = norm
        self.feature_norm = None
        self.message_norm = None

        if norm == "batch":
            self.feature_norm = BatchNorm(self.hidden_irreps)
            self.message_norm = BatchNorm(self.hidden_irreps)
        elif norm == "instance":
            self.feature_norm = InstanceNorm(self.hidden_irreps)

    def forward(
        self,
        x,
        edge_index,
        edge_attr,
        node_attr,
        batch,
        additional_message_features=None,
    ):
        """Propagate messages along edges"""
        x = self.propagate(
            edge_index,
            x=x,
            node_attr=node_attr,
            edge_attr=edge_attr,
            additional_message_features=additional_message_features,
        )
        # Normalise features
        if self.feature_norm:
            if self.norm == "batch":
                x = self.feature_norm(x)
            elif self.norm == "instance":
                x = self.feature_norm(x, batch)
        return x

    def message(self, x_i, x_j, edge_attr, additional_message_features):
        """Create messages"""
        """
            edge_attr: a_i_j
            additional_message_features: 
                any additional edge attribute that doesnt steer the kernel
                its the radial part 
                in n-body its an absolute distance between nodes and multiplier of charges between nodes 
        """
        if additional_message_features is None:
            input = torch.cat((x_i, x_j), dim=-1)
        else:
            # this is the "h_ij", this concatenates input for messages between two neighbouring nodes
            input = torch.cat((x_i, x_j, additional_message_features), dim=-1)

        message = self.message_layer_1(input, edge_attr)
        message = self.message_layer_2(message, edge_attr)

        if self.message_norm:
            message = self.message_norm(message)
        return message

    def update(self, message, x, node_attr):
        """Update note features"""
        """
            mesasge:
                output from message passing layer (summed messages from all neighbours)
            x:
                current node features (previous update if there are more MP layers)
            node_attr: 
                a_i
                this attribute is embedded in SH
                it steers the node update
                in n-body problem its calculated by summing all neighbouring edge_attr (rel. dist. in SH) and then adding nodal features embedded in SH   
                there is no radial part???
        """
        input = torch.cat((x, message), dim=-1)
        update = self.update_layer_1(input, node_attr)
        update = self.update_layer_2(update, node_attr)
        x += update  # Residual connection.. its residual because instead of just returning the new node value it adds to the previous one (previous values still affect the updates)
        return x
