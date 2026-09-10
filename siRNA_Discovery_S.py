# --- START OF FILE siRNA_Discovery_S.py (Corrected for MyDataset Error) ---

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, roc_auc_score
import scipy.stats
import json
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.sparse as sp
import utils
from tensorflow.keras import layers, Model, optimizers, callbacks
from spektral.layers import GraphSageConv
from spektral.data import Graph, Dataset
import math  # Added this line

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # This line forces TensorFlow to use the CPU

params = json.load(open("siRNA_param.json", "r"))

score_PCC = []
score_SPCC = []
score_mse = []
score_auc = []

for n in range(3):

    """
    Read File
    """
    data_train = pd.read_csv("siRNA_split_datasets/split" + str(n) + "/train.csv")
    data_dev = pd.read_csv("siRNA_split_datasets/split" + str(n) + "/dev.csv")
    data_test = pd.read_csv("siRNA_split_datasets/split" + str(n) + "/test.csv")

    data_train["siRNA_seq"] = data_train["siRNA_seq"].replace("U", "T", regex=True)
    data_dev["siRNA_seq"] = data_dev["siRNA_seq"].replace("U", "T", regex=True)
    data_test["siRNA_seq"] = data_test["siRNA_seq"].replace("U", "T", regex=True)

    data = pd.concat([data_train, data_dev, data_test], axis=0)

    """
    Feature processing
    """
    # one-hot
    sirna_onehot = [
        utils.obtain_one_hot_feature_for_one_sequence_1(seq, params["sirna_length"])
        for seq in data["siRNA_seq"]
    ]
    sirna_onehot = pd.DataFrame(sirna_onehot, index=list(data["siRNA"]))

    mrna_onehot_temp = data.loc[:, ["mRNA", "mRNA_seq"]]
    mrna_onehot_temp = mrna_onehot_temp.drop_duplicates(subset="mRNA")

    mrna_onehot = [
        utils.obtain_one_hot_feature_for_one_sequence_1(seq, params["max_mrna_len"])
        for seq in mrna_onehot_temp["mRNA_seq"]
    ]
    mrna_onehot = pd.DataFrame(mrna_onehot, index=list(mrna_onehot_temp["mRNA"]))

    # Positional encoding
    trans_table = str.maketrans("ATCG", "TAGC")
    data["match_pos"] = [
        seq[::-1].upper().translate(trans_table) for seq in data["siRNA_seq"]
    ]
    data["match_pos"] = data.apply(
        lambda row: row["mRNA_seq"].index(row["match_pos"]), axis=1
    )

    sirna_pos_encoding = [
        utils.get_pos_embedding_sequence(num, params["sirna_length"], params["dmodel"])
        for num in data["match_pos"]
    ]
    sirna_pos_encoding = pd.DataFrame(sirna_pos_encoding, index=list(data["siRNA"]))

    # Thermodynamics
    sirna_thermo_feat = [
        utils.cal_thermo_feature(seq.replace("T", "U")) for seq in data["siRNA_seq"]
    ]
    sirna_thermo_feat = pd.DataFrame(sirna_thermo_feat).reset_index(drop=True)
    sirna_thermo_feat = pd.concat(
        [
            data["siRNA"].reset_index(drop=True),
            data["mRNA"].reset_index(drop=True),
            sirna_thermo_feat,
        ],
        axis=1,
    )
    sirna_thermo_feat["index"] = (
        sirna_thermo_feat["siRNA"] + "_" + sirna_thermo_feat["mRNA"]
    )
    sirna_thermo_feat = sirna_thermo_feat.set_index("index").drop(
        columns=["siRNA", "mRNA"]
    )

    # Co-fold features
    con_feat = pd.read_csv(
        "siRNA_split_preprocess/con_matrix.txt", header=None, index_col=0
    )
    con_feat = con_feat.reindex(sirna_thermo_feat.index)

    # sel-fold features
    sirna_sfold_feat = pd.read_csv(
        "siRNA_split_preprocess/self_siRNA_matrix.txt", header=None, index_col=0
    )
    sirna_sfold_feat = sirna_sfold_feat.reindex(sirna_onehot.index)

    mrna_sfold_feat = pd.read_csv(
        "siRNA_split_preprocess/self_mRNA_matrix.txt", header=None, index_col=0
    )
    mrna_sfold_feat = mrna_sfold_feat.reindex(mrna_onehot.index)

    # AGO2
    sirna_ago = pd.read_csv("RNA_AGO2/siRNA_AGO2.csv", index_col=0)
    sirna_ago = sirna_ago.reindex(sirna_onehot.index)

    mrna_ago = pd.read_csv("RNA_AGO2/mRNA_AGO2.csv", index_col=0)
    mrna_ago = mrna_ago.reindex(mrna_onehot.index)

    # GC percentage
    sirna_GC = [utils.countGC(seq) for seq in data["siRNA_seq"]]
    sirna_GC = pd.DataFrame(sirna_GC, index=list(data["siRNA"]))

    mrna_GC = [utils.countGC(seq) for seq in mrna_onehot_temp["mRNA_seq"]]
    mrna_GC = pd.DataFrame(mrna_GC, index=list(mrna_onehot_temp["mRNA"]))

    # k-mers
    sirna_1_mer = pd.DataFrame([utils.single_freq(seq) for seq in data["siRNA_seq"]])
    sirna_2_mers = pd.DataFrame([utils.double_freq(seq) for seq in data["siRNA_seq"]])
    sirna_3_mers = pd.DataFrame([utils.triple_freq(seq) for seq in data["siRNA_seq"]])
    sirna_4_mers = pd.DataFrame(
        [utils.quadruple_freq(seq) for seq in data["siRNA_seq"]]
    )
    sirna_5_mers = pd.DataFrame(
        [utils.quintuple_freq(seq) for seq in data["siRNA_seq"]]
    )
    sirna_k_mers = pd.concat(
        [sirna_1_mer, sirna_2_mers, sirna_3_mers, sirna_4_mers, sirna_5_mers], axis=1
    )
    sirna_k_mers.index = data["siRNA"]

    # siRNA rules codes
    sirna_pos_scores = [utils.rules_scores(seq) for seq in data["siRNA_seq"]]
    sirna_pos_scores = pd.DataFrame(sirna_pos_scores, index=list(data["siRNA"]))

    """
    The features of GNN nodes
    """
    sirna_pd = pd.concat(
        [
            sirna_onehot,
            sirna_sfold_feat,
            sirna_ago,
            sirna_GC,
            sirna_k_mers,
            sirna_pos_scores,
        ],
        axis=1,
    )

    mrna_pd = pd.concat([mrna_onehot, mrna_sfold_feat, mrna_ago, mrna_GC], axis=1)

    source = data["siRNA"] + "_" + data["mRNA"]
    target_siRNA = data["siRNA"]
    target_mRNA = data["mRNA"]

    all_my_edges1 = pd.DataFrame({"source": source, "target": target_siRNA})
    all_my_edges2 = pd.DataFrame({"source": source, "target": target_mRNA})

    all_my_edges = pd.concat([all_my_edges1, all_my_edges2], ignore_index=True, axis=0)

    sirna_pos_encoding.index = sirna_thermo_feat.index

    interaction_pd = pd.concat(
        [sirna_thermo_feat, con_feat, sirna_pos_encoding], axis=1
    )

    """
    Model training with Spektral
    """
    # 1. Graph Construction for Spektral
    unique_sirna_ids = sirna_pd.index.unique()
    unique_mrna_ids = mrna_pd.index.unique()
    unique_interaction_ids = interaction_pd.index.unique()

    all_node_ids = (
        list(unique_sirna_ids) + list(unique_mrna_ids) + list(unique_interaction_ids)
    )
    node_id_to_idx = {node_id: i for i, node_id in enumerate(all_node_ids)}

    num_sirna_nodes = len(unique_sirna_ids)
    num_mrna_nodes = len(unique_mrna_ids)
    num_interaction_nodes = len(unique_interaction_ids)
    total_nodes = num_sirna_nodes + num_mrna_nodes + num_interaction_nodes

    max_feat_dim = max(sirna_pd.shape[1], mrna_pd.shape[1], interaction_pd.shape[1])

    X = np.zeros((total_nodes, max_feat_dim + 3))  # +3 for one-hot node type

    for node_id, idx in node_id_to_idx.items():
        if node_id in sirna_pd.index:
            feats = sirna_pd.loc[node_id].values
            X[idx, : len(feats)] = feats
            X[idx, -3] = 1  # One-hot for siRNA
        elif node_id in mrna_pd.index:
            feats = mrna_pd.loc[node_id].values
            X[idx, : len(feats)] = feats
            X[idx, -2] = 1  # One-hot for mRNA
        elif node_id in interaction_pd.index:
            feats = interaction_pd.loc[node_id].values
            X[idx, : len(feats)] = feats
            X[idx, -1] = 1  # One-hot for interaction

    rows, cols = [], []
    for _, row in all_my_edges.iterrows():
        src_idx = node_id_to_idx[row["source"]]
        tgt_idx = node_id_to_idx[row["target"]]
        rows.append(src_idx)
        cols.append(tgt_idx)
        rows.append(
            tgt_idx
        )  # Assuming undirected graph for simplicity in GNN aggregation
        cols.append(src_idx)

    # ... (previous code) ...

    A = sp.csr_matrix(
        (np.ones(len(rows)), (rows, cols)),
        shape=(total_nodes, total_nodes),
        dtype=np.float32,
    )
    # Convert the scipy sparse matrix to a TensorFlow SparseTensor
    A = tf.sparse.SparseTensor(
        indices=np.array(A.nonzero()).T,
        values=A.data,
        dense_shape=A.shape,
    )
    A = tf.cast(A, tf.float32)  # Ensure the sparse tensor also has float32 dtype

    # Corrected MyDataset class
    class MyDataset(Dataset):
        def __init__(self, x_data, a_data, y_data=None, **kwargs):
            self._x_data = x_data
            self._a_data = a_data
            self._y_data = y_data
            super().__init__(**kwargs)

        def read(self):
            # Ensure x is float32 as well, if it's not already
            return [
                Graph(
                    x=tf.cast(self._x_data, tf.float32), a=self._a_data, y=self._y_data
                )
            ]

    # Instantiate the corrected Spektral graph object
    spektral_graph = MyDataset(x_data=X, a_data=A)

    # ... (rest of the code) ...

    # 2. HinSAGE-like Model with Spektral
    class HinSAGESpektral(Model):
        def __init__(
            self, layer_sizes, dropout, n_output_feats, activation=tf.nn.relu, **kwargs
        ):
            super().__init__(**kwargs)
            self.conv_layers = []
            for i, size in enumerate(layer_sizes):
                self.conv_layers.append(
                    GraphSageConv(
                        size,
                        activation=activation,
                        dropout_rate=dropout,
                        name=f"hinsage_conv_{i}",
                    )
                )

        def call(self, inputs):
            x, a = inputs
            for conv_layer in self.conv_layers:
                x = conv_layer([x, a])
            return x

    class SelectInteractionNodes(layers.Layer):
        def __init__(self, interaction_node_indices, **kwargs):
            super().__init__(**kwargs)
            self.interaction_node_indices = tf.constant(
                interaction_node_indices, dtype=tf.int32
            )

        def call(self, inputs):
            return tf.gather(inputs, self.interaction_node_indices, axis=0)

    # Create the GNN model instance
    hinsage_gnn_model = HinSAGESpektral(
        layer_sizes=params["hinsage_layer_sizes"],
        dropout=params["dropout"],
        n_output_feats=params["hinsage_layer_sizes"][-1],
    )

    # Define inputs for the full Keras model
    input_x = layers.Input(shape=(X.shape[1],), name="node_features")
    input_a = layers.Input(
        shape=(total_nodes,), sparse=True, name="adj_matrix", dtype=tf.float32
    )

    gnn_output = hinsage_gnn_model([input_x, input_a])

    interaction_node_indices_in_full_graph = [
        node_id_to_idx[nid] for nid in unique_interaction_ids
    ]
    selected_interaction_features = SelectInteractionNodes(
        interaction_node_indices_in_full_graph
    )(gnn_output)

    prediction_output = layers.Dense(units=1, name="final_prediction")(
        selected_interaction_features
    )

    model = Model(inputs=[input_x, input_a], outputs=prediction_output)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=params["lr"]), loss=params["loss"]
    )

    class SpektralGenerator(tf.keras.utils.Sequence):
        def __init__(
            self, graph_data, interaction_ids, efficacy_df, node_id_to_idx, batch_size
        ):
            self.x = graph_data.x
            self.a = graph_data.a
            self.interaction_ids = np.array(interaction_ids)
            self.efficacy_df = efficacy_df
            self.node_id_to_idx = node_id_to_idx
            self.batch_size = batch_size
            self.on_epoch_end()

        def __len__(self):
            return math.ceil(len(self.interaction_ids) / self.batch_size)

        def __getitem__(self, index):
            batch_interaction_ids = self.interaction_ids[
                index * self.batch_size : (index + 1) * self.batch_size
            ]

            batch_y = self.efficacy_df.loc[batch_interaction_ids].values

            return (self.x, self.a), batch_y.reshape(-1, 1)

        def on_epoch_end(self):
            np.random.shuffle(self.interaction_ids)

    train_interaction = pd.DataFrame(
        data_train["efficacy"].values,
        index=data_train["siRNA"] + "_" + data_train["mRNA"],
    )
    dev_interaction = pd.DataFrame(
        data_dev["efficacy"].values, index=data_dev["siRNA"] + "_" + data_dev["mRNA"]
    )
    test_interaction = pd.DataFrame(
        data_test["efficacy"].values, index=data_test["siRNA"] + "_" + data_test["mRNA"]
    )

    train_gen = SpektralGenerator(
        spektral_graph.read()[0],
        list(train_interaction.index),
        train_interaction,
        node_id_to_idx,
        params["batch_size"],
    )
    dev_gen = SpektralGenerator(
        spektral_graph.read()[0],
        list(dev_interaction.index),
        dev_interaction,
        node_id_to_idx,
        params["batch_size"],
    )
    test_gen_for_metrics = SpektralGenerator(
        spektral_graph.read()[0],
        list(test_interaction.index),
        test_interaction,
        node_id_to_idx,
        params["batch_size"],
    )

    history = model.fit(
        train_gen, epochs=params["epochs"], validation_data=dev_gen, verbose=2
    )

    """
    Evaluate on the testset and Calculate the metrics with Spektral model
    """
    full_gnn_output_features = hinsage_gnn_model.predict(
        (spektral_graph.read()[0].x, spektral_graph.read()[0].a)
    )

    all_interaction_node_features = SelectInteractionNodes(
        interaction_node_indices_in_full_graph
    )(full_gnn_output_features)

    test_predictions_raw = model.get_layer("final_prediction")(
        all_interaction_node_features
    ).numpy()
    test_predictions_raw = np.squeeze(test_predictions_raw)

    predictions_series = pd.Series(test_predictions_raw, index=unique_interaction_ids)

    test = predictions_series.reindex(test_interaction.index).values
    test_true = test_interaction.values.ravel()

    # PCC
    pearson = scipy.stats.pearsonr(test_true, test)
    score_PCC.append(pearson[0])
    print(f"PCC: {pearson[0]:.4f}")

    # SPCC
    spearman = scipy.stats.spearmanr(test_true, test)
    score_SPCC.append(spearman[0])
    print(f"SPCC: {spearman[0]:.4f}")

    # MSE
    mse_run = mean_squared_error(test_true, test)
    score_mse.append(mse_run)
    print(f"MSE: {mse_run:.4f}")

    # AUC
    binary_pred = (test_true > 0.7).astype(int)
    auc = roc_auc_score(binary_pred, test)
    score_auc.append(auc)
    print(f"AUC: {auc:.4f}")

    print(f"{n} finished!")


print("\n--- Overall Scores ---")
print(f"Overall PCC score = {np.mean(score_PCC):.4f}")
print(f"Overall SPCC score = {np.mean(score_SPCC):.4f}")
print(f"Overall MSE score = {np.mean(score_mse):.4f}")
print(f"Overall AUC score = {np.mean(score_auc):.4f}")
