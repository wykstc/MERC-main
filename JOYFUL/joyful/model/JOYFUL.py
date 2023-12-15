import torch
import torch.nn as nn

from .SeqContext import SeqContext
from .GNN import GNN
from .Classifier import Classifier
from .functions import batch_graphify
import joyful

log = joyful.utils.get_logger()


class JOYFUL(nn.Module):
    def __init__(self, args):
        super(JOYFUL, self).__init__()
        u_dim = 100
        if args.rnn == "transformer":
            g_dim = args.hidden_size
        else:
            g_dim = 200
        h1_dim = args.hidden_size
        h2_dim = args.hidden_size
        hc_dim = args.hidden_size
        dataset_label_dict = {
            "iemocap": {"hap": 0, "sad": 1, "neu": 2, "ang": 3, "exc": 4, "fru": 5},
            "iemocap_4": {"hap": 0, "sad": 1, "neu": 2, "ang": 3},
            "mosei": {"Negative": 0, "Positive": 1},
            "meld": {"Neutral": 0, "Surprise": 1, "Fear": 2, "Sadness": 3, "Joy": 4, "Disgust": 5, "Angry": 6}
        }

        dataset_speaker_dict = {
            "iemocap": 2,
            "iemocap_4": 2,
            "mosei": 1,
            "meld": 9
        }

        if args.dataset and args.emotion == "multilabel":
            dataset_label_dict["mosei"] = {
                "happiness": 0,
                "sadness": 1,
                "anger": 2,
                "surprise": 3,
                "disgust": 4,
                "fear": 5,
            }


        tag_size = len(dataset_label_dict[args.dataset])
        args.n_speakers = dataset_speaker_dict[args.dataset]
        self.concat_gin_gout = args.concat_gin_gout
        self.cl_loss_weight = args.cl_loss_weight

        self.wp = args.wp
        self.wf = args.wf
        self.device = args.device

        self.rnn = SeqContext(u_dim, g_dim, args)
        self.gcl = GNN(g_dim, h1_dim, h2_dim, args)
        if args.concat_gin_gout:
            self.clf = Classifier(
                g_dim + h2_dim * args.gnn_nheads, hc_dim, tag_size, args
            )
        else:
            self.clf = Classifier(h2_dim * args.gnn_nheads, hc_dim, tag_size, args)

        edge_type_to_idx = {}
        for j in range(args.n_speakers):
            for k in range(args.n_speakers):
                edge_type_to_idx[str(j) + str(k) + "0"] = len(edge_type_to_idx)
                edge_type_to_idx[str(j) + str(k) + "1"] = len(edge_type_to_idx)
        self.edge_type_to_idx = edge_type_to_idx
        log.debug(self.edge_type_to_idx)

    def get_rep(self, data=None, whetherT=None):
        node_features = self.rnn(data["text_len_tensor"], data["input_tensor"])


        features, edge_index, edge_type, edge_index_lengths = batch_graphify(
            node_features,
            data["text_len_tensor"],
            data["speaker_tensor"],
            self.wp,
            self.wf,
            self.edge_type_to_idx,
            self.device,
        )

        graph_out, cl_loss = self.gcl(features, edge_index, edge_type, whetherT)
        return graph_out, features, cl_loss

    def forward(self, data, whetherT):
        graph_out, features, cl_loss = self.get_rep(data,whetherT)
        if self.concat_gin_gout:
            out = self.clf(
                torch.cat([features, graph_out], dim=-1), data["text_len_tensor"]
            )
        else:
            out = self.clf(graph_out, data["text_len_tensor"])

        return out

    def get_loss(self, data, whetherT):
        graph_out, features, cl_loss = self.get_rep(data, whetherT)
        if self.concat_gin_gout:
            loss = self.clf.get_loss(
                torch.cat([features, graph_out], dim=-1),
                data["label_tensor"],
                data["text_len_tensor"],
            )
        else:
            loss = self.clf.get_loss(
                graph_out, data["label_tensor"], data["text_len_tensor"]
            )
        return loss + self.cl_loss_weight*cl_loss
