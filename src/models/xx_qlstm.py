from sklearn.inspection import partial_dependence
import torch
import torch.nn as nn
import pytorch_lightning as pl
import pennylane as qml
import math


class XXQLSTM(pl.LightningModule):
    def __init__(
        self,
        config,
    ):
        super(XXQLSTM, self).__init__()

        ## CONFIG
        self.config = config
        self.config_model = config["xx-QLSTM"]
        self.input_dim = self.config_model["input_dim"]
        self.hidden_dim = self.config_model["hidden_dim"]
        self.concat_size = self.input_dim + self.hidden_dim
        self.n_qubits = self.config_model["n_qubits"]
        self.depth = self.config_model["depth"]
        self.backend = config[
            "backend"
        ]  # "default.qubit", "qiskit.basicaer", "qiskit.ibm"
        self.batch_first = config["batch_first"]
        self.diff_method = self.config_model["diff_method"]
        self.encoding = self.config_model["encoding"]
        self.vqc_config = self.config_model["vqc"]
        self.dropout_rate = self.config_model["dropout"]
        self.four_linear_before_vqc = self.config_model["four_linear_before_vqc"]
        self.combine_linear_after_vqc = self.config_model["combine_linear_after_vqc"]

        self.setup()

    def forward(self, x, init_states=None):
        """
        x.shape is (batch_size, seq_length, feature_size)
        recurrent_activation -> sigmoid
        activation -> tanh
        """
        if self.batch_first is True:
            batch_size, seq_length, features_size = x.size()
        else:
            seq_length, batch_size, features_size = x.size()

        hidden_seq = []
        if init_states is None:
            h_t = torch.zeros(
                batch_size, self.hidden_dim, device=self.device
            )  # hidden state (output)
            c_t = torch.zeros(
                batch_size, self.hidden_dim, device=self.device
            )  # cell state
        else:
            # for now we ignore the fact that in PyTorch you can stack multiple RNNs
            # so we take only the first elements of the init_states tuple init_states[0][0], init_states[1][0]
            h_t, c_t = init_states
            h_t = h_t[0]
            c_t = c_t[0]

        for t in range(seq_length):
            # get features from the t-th element in seq, for all entries in the batch
            x_t = x[:, t, :] if self.batch_first is True else x[t]
            # Concatenate input and hidden state
            v_t = torch.cat((h_t, x_t), dim=1)

            # match qubit dimension
            if self.four_linear_before_vqc:
                y_t_forget = self.clayer_in_forget(v_t)
                y_t_input = self.clayer_in_input(v_t)
                y_t_update = self.clayer_in_update(v_t)
                y_t_output = self.clayer_in_output(v_t)
                f_t_vqc = self.vqc_forget(y_t_forget)  # forget block
                i_t_vqc = self.vqc_input(y_t_input)
                g_t_vqc = self.vqc_update(y_t_update)
                o_t_vqc = self.vqc_output(y_t_output)
            else:
                y_t = self.clayer_in(v_t)
                f_t_vqc = self.vqc_forget(y_t)  # forget block
                i_t_vqc = self.vqc_input(y_t)  # input block
                g_t_vqc = self.vqc_update(y_t)  # update block
                o_t_vqc = self.vqc_output(y_t)  # output block

            if self.combine_linear_after_vqc:
                f_t = torch.sigmoid(self.clayer_out(f_t_vqc))
                i_t = torch.sigmoid(self.clayer_out(i_t_vqc))
                g_t = torch.tanh(self.clayer_out(g_t_vqc))
                o_t = torch.sigmoid(self.clayer_out(o_t_vqc))
            else:
                f_t = torch.sigmoid(self.clayer_out_forget(f_t_vqc))
                i_t = torch.sigmoid(self.clayer_out_input(i_t_vqc))
                g_t = torch.tanh(self.clayer_out_update(g_t_vqc))
                o_t = torch.sigmoid(self.clayer_out_output(o_t_vqc))

            c_t = (f_t * c_t) + (i_t * g_t)
            h_t = o_t * torch.tanh(c_t)

            hidden_seq.append(h_t.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0)
        # hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (h_t, c_t)

    def setup(self):

        part_names = ["forget", "input", "update", "output"]
        self.wires_list = [
            [f"wire_{part}_{i}" for i in range(self.n_qubits)] for part in part_names
        ]
        self.devs = [qml.device(self.backend, wires=wires) for wires in self.wires_list]

        self.vqc_forget = self.vqc_setup(self.devs[0], self.wires_list[0])
        self.vqc_input = self.vqc_setup(self.devs[1], self.wires_list[1])
        self.vqc_update = self.vqc_setup(self.devs[2], self.wires_list[2])
        self.vqc_output = self.vqc_setup(self.devs[3], self.wires_list[3])

        # EMBEDDING
        self.dropout = torch.nn.Dropout(self.dropout_rate)
        if not self.combine_linear_after_vqc:
            self.clayer_out_forget = torch.nn.Linear(self.n_qubits, self.hidden_dim)
            self.clayer_out_input = torch.nn.Linear(self.n_qubits, self.hidden_dim)
            self.clayer_out_update = torch.nn.Linear(self.n_qubits, self.hidden_dim)
            self.clayer_out_output = torch.nn.Linear(self.n_qubits, self.hidden_dim)
        else:
            self.clayer_out = torch.nn.Linear(self.n_qubits, self.hidden_dim)
        if self.four_linear_before_vqc:
            self.clayer_in_forget = torch.nn.Linear(self.concat_size, self.n_qubits)
            self.clayer_in_input = torch.nn.Linear(self.concat_size, self.n_qubits)
            self.clayer_in_update = torch.nn.Linear(self.concat_size, self.n_qubits)
            self.clayer_in_output = torch.nn.Linear(self.concat_size, self.n_qubits)
        else:
            self.clayer_in = torch.nn.Linear(self.concat_size, self.n_qubits)

    def vqc_setup(self, dev, wires):
        @qml.qnode(dev, diff_method="best", interface="torch")
        def circuit(inputs, weights):
            for i, feat in enumerate(inputs):
                # encoding
                if self.encoding == "original":
                    qml.Hadamard(wires=wires[i])
                    qml.RY(math.atan(feat), wires=wires[i])
                    qml.RZ(math.atan(feat**2), wires=wires[i])
                elif self.encoding == "No-H":
                    qml.RY(math.atan(feat), wires=wires[i])
                    qml.RZ(math.atan(feat**2), wires=wires[i])
                elif self.encoding == "No-Square":
                    qml.Hadamard(wires=wires[i])
                    qml.RX(math.atan(feat), wires=wires[i])
                elif self.encoding == "arcsin-arccos":
                    feat = min(1, max(feat, -1))
                    qml.RY(math.asin(feat), wires=wires[i])
                    qml.RZ(math.acos(feat), wires=wires[i])
                elif self.encoding == "5":
                    qml.Hadamard(wires=wires[i])
                    qml.RX(math.atan(feat), wires=wires[i])
                    qml.RZ(math.atan(feat**2), wires=wires[i])

            assert len(weights.shape) == 3
            n_qlayer = weights.shape[0]
            n_wire = weights.shape[1]

            # vqc
            weight_shapes = None
            for i in range(n_qlayer):
                if self.vqc_config == "original":
                    assert weights.shape[2] == 3
                    for j in range(n_wire):
                        qml.CNOT([wires[j], wires[(j + 1) % n_wire]])
                    for j in range(n_wire):
                        qml.CNOT([wires[j], wires[(j + 2) % n_wire]])
                    for j in range(n_wire):
                        qml.Rot(*weights[i][j], wires=wires[j])
                elif self.vqc_config == "5":
                    remaining = [n_wire - 1] * n_wire
                    assert weights.shape[2] == n_wire - 1 + 2
                    for j in reversed(range(n_wire)):
                        for k in reversed(range(n_wire)):
                            if k != j:
                                qml.CRZ(
                                    phi=weights[i][k][remaining[k]],
                                    wires=[wires[j], wires[k]],
                                )
                                remaining[k] -= 1
                    for j in range(n_wire):
                        qml.RX(phi=weights[i][j][-2], wires=wires[j])
                        qml.RZ(phi=weights[i][j][-1], wires=wires[j])
                elif self.vqc_config == "10":
                    for j in range(n_wire):
                        qml.CNOT([wires[j], wires[(j + 1) % n_wire]])
                    for j in range(n_wire):
                        qml.RY(weights[i][j][0], wires=wires[j])
                elif self.vqc_config == "18":
                    for j in range(n_wire):
                        qml.CRZ(
                            weights[i][j][0], wires=[wires[j], wires[(j + 1) % n_wire]]
                        )

            return [qml.expval(qml.PauliZ(wires=w)) for w in wires]

        if self.vqc_config == "original":
            weight_shapes = {"weights": (self.depth, self.n_qubits, 3)}
        elif self.vqc_config == "5":
            weight_shapes = {
                "weights": (self.depth, self.n_qubits, self.n_qubits - 1 + 2)
            }
        elif self.vqc_config == "10":
            weight_shapes = {"weights": (self.depth, self.n_qubits, 1)}
        elif self.vqc_config == "18":
            weight_shapes = {"weights": (self.depth, self.n_qubits, 1)}
        return qml.qnn.TorchLayer(circuit, weight_shapes)
