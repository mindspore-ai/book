from mindspore.nn.layer import Cell
from mindspore.ops import operations
from mindspore.nn.layer.core import Dense
from mindspore.nn.layer.activation import ReLU
from mindspore.application.gnn import initialize_embedded_graph
from mindspore.application.gnn.base import get_feature, get_neighbor, get_label
from mindspore.ops.nn_ops import Momentum
from mindspore.core.parameter import Parameter
from mindspore.application.gnn.base import NetWithLossClass, GradWrap

class GCNAggregator(Cell):
    def __init__(self, in_dim, out_dim):
          super(GCNAggregator, self).__init__()
          self.add = operations.TensorAdd()
          self.div = operations.TensorDiv()
          self.spmm = operations.SparseDenseMatmul()
          self.fc = Dense(in_dim, out_dim)
          self.relu = ReLU()
    def construct(self, adj, node_emb, neighbor_emb):
          agg_emb = self.spmm(adj[0], adj[1], adj[2], neighbor_emb)
          agg_emb = self.add(node_emb, agg_emb)
          agg_emb = self.div(agg_emb, adj[3])
          agg_emb = self.fc(agg_emb)
          agg_emb = self.relu(agg_emb)
          return agg_emb

class SingleLayerGCN(Cell):
    def __init__(self, in_dim, out_dim, num_classes):
        super(SingleLayerGCN, self).__init__()
        self.aggregator = GCNAggregator(in_dim, out_dim)
        self.output_layer = Dense(out_dim, num_classes)
    def construct(self, adj, node_feature, neighbor_feature ):
        embeddings = self.aggregator(adj, node_feature, neighbor_feature)
        output = self.output_layer(embeddings)
        return output

def GCNTrainer(in_dim, out_dim, num_classes,num_epoch, graph_data):
    input_node, neighbor_node, node_feature, neighbor_feature, labels = graph_data
    network = SingleLayerGCN(in_dim, out_dim, num_classes)
    loss_network = NetWithLossClass(network)
    train_net = GradWrap(loss_network)
    train_net.train(True)
    parameters = train_net.weights
    momentum = Momentum()
    lr_v = Parameter(0.01, name="learning_rate")
    momen_v = Parameter(0.01, name="momentum")
    for _ in range(num_epoch):
        grads = train_net.construct(adj_list[0], node_feature, neighbor_feature, labels)
        accumulations = parameters.clone(prefix='moments')
        for i, element in enumerate(grads):
            updated = momentum(element, accumulations[i], parameters[i], lr_v, momen_v)
            parameters[i].set_parameter_data(updated)

initilize_embedded_graph(GRAPH_DIR)
neighbor_node, adj_list = get_neighbor(input_node, k_hop)
node_feature = get_feature(input_node)
neighbor_feature = get_feature(neighbor_node)
labels = get_label(input_node)
graph_data = [input_node, neighbor_node, node_feature, neighbor_feature, labels]
in_dim = IN_DIM
out_dim = OUT_DIM
num_classes = CLASS_NUM
num_epoch = EPOCH_NUM
GCNTrainer(in_dim, out_dim, num_classes,num_epoch, graph_data)
