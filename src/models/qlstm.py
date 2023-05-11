import torch
import torch.nn as nn
import pytorch_lightning as pl
import math
import pennylane as qml


class QLSTM(pl.LightningModule):
    def __init__(self, config):
        super(QLSTM, self).__init__()

        self.config = config
        self.config_qlstm = config["QLSTM"]
        self.input_dim = self.config_qlstm["input_dim"]
        self.hidden_dim = self.config_qlstm["hidden_dim"]
        self.concat_size = self.input_dim + self.hidden_dim
        self.n_qubits = self.config_qlstm["n_qubits"]
        self.depth = self.config_qlstm["depth"]
        self.backend = config[
            "backend"
        ]  # "default.qubit", "qiskit.basicaer", "qiskit.ibm"
        self.batch_first = config["batch_first"]
        self.diff_method = self.config_qlstm["diff_method"]
        self.encoding = self.config_qlstm["encoding"]
        self.vqc_config = self.config_qlstm["vqc"]

        self.setup()

    def forward(self, x, init_states=None):
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

            f_t = torch.sigmoid(self.vqc1(v_t))  # forget block
            i_t = torch.sigmoid(self.vqc2(v_t))  # input block
            c_tile_t = torch.tanh(self.vqc3(v_t))  # update block
            c_t = (f_t * c_t) + (i_t * c_tile_t)
            o_t = torch.sigmoid(self.vqc4(v_t))  # output block
            h_t = self.vqc5(o_t * torch.tanh(c_t))
            y_t = self.vqc6(o_t * torch.tanh(c_t))

        return y_t

    def setup(self):

        self.wires_list = [
            [f"vqc_{j}_{i}" for i in range(self.n_qubits)] for j in range(6)
        ]
        self.devs = [qml.device(self.backend, wires=wires) for wires in self.wires_list]

        self.measure_wires_inside = slice(0, self.hidden_dim)
        self.measure_wires_output = 0
        self.vqc1 = self.vqc_setup(
            self.devs[0], self.wires_list[0], self.measure_wires_inside
        )
        self.vqc2 = self.vqc_setup(
            self.devs[1], self.wires_list[1], self.measure_wires_inside
        )
        self.vqc3 = self.vqc_setup(
            self.devs[2], self.wires_list[2], self.measure_wires_inside
        )
        self.vqc4 = self.vqc_setup(
            self.devs[3], self.wires_list[3], self.measure_wires_inside
        )
        self.vqc5 = self.vqc_setup(
            self.devs[4], self.wires_list[4], self.measure_wires_inside
        )
        self.vqc6 = self.vqc_setup(
            self.devs[5], self.wires_list[5], self.measure_wires_output
        )

    def vqc_setup(self, dev, wires, measure_wire_indices):
        measure_wires = wires[measure_wire_indices]
        if not isinstance(measure_wires, list):
            measure_wires = [measure_wires]

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
                    qml.RY(math.asin(feat), wires=wires[i])
                    qml.RZ(math.acos(feat), wires=wires[i])

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

            return [qml.expval(qml.PauliZ(wires=w)) for w in measure_wires]

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
        print(
            f"weight_shapes = (depth, n_qubits, 3) = ({self.depth}, {self.n_qubits}, 3)"
        )
        return qml.qnn.TorchLayer(circuit, weight_shapes)
