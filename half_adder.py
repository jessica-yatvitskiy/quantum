#Implements a half-adder circuit

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from numpy import pi

#Need four qubits (two inputs, an "add" qubit, and a "carry" qubit)
#qubits initialized in |0> state: cos(0/2)*|0> + sin(0/2)*e^(0i)*|1> = [1 0]^T = |0>
qreg_q = QuantumRegister(4, 'q') 

#Need four classical bits (to store values after measurement)
creg_c = ClassicalRegister(4, 'c') 

#Creates circuit that will act on these qubits and bits
circuit = QuantumCircuit(qreg_q, creg_c) 

circuit.h(qreg_q[0]) #have a hadamard gate act on qubit 0 --> sets 
circuit.h(qreg_q[1])
circuit.cx(qreg_q[0], qreg_q[2])
circuit.cx(qreg_q[1], qreg_q[2])
circuit.ccx(qreg_q[0], qreg_q[1], qreg_q[3])
