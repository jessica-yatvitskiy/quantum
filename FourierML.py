import sys
from pennylane import numpy as np
import pennylane as qml


def generating_fourier_state(n_qubits, m):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(angles):
        for i in range(n_qubits):
            qml.Hadamard(wires = i)
            qml.RZ(angles[i], wires = i)

        qml.adjoint(qml.QFT)(wires=range(n_qubits))
        return qml.probs(wires=range(n_qubits))

    def error(angles):
        probs = circuit(angles)
        return np.sum(probs**2) + 1 - 2*probs[m]

    opt = qml.AdamOptimizer(stepsize=0.8)
    epochs = 1000

    angles = np.zeros(n_qubits, requires_grad=True)

    for epoch in range(epochs):
        angles = opt.step(error, angles)
        angles = np.clip(opt.step(error, angles), -2 * np.pi, 2 * np.pi)

    return circuit, angles

