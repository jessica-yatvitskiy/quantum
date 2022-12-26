#Implements a full_adder circuit

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from numpy import pi

num_qubits=input()
qreg_q = QuantumRegister(num_qubits, 'q') 
creg_c = ClassicalRegister(num_qubits, 'c')
circuit = QuantumCircuit(qreg_q, creg_c) 
qubit_array=[]
for i in range(0,input()-2):
  circuit.h(qreg_q[i])
  circuit.cx(qreg_q[i], qreg_q[i+2]) 
  circuit.cx(qreg_q[i+1], qreg_q[i+2]) 
  circuit.ccx(qreg_q[i], qreg_q[i+1], qreg_q[i+3])

