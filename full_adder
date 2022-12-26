#Implements a full_adder circuit

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from numpy import pi

num_qubits=input()
qreg_q = QuantumRegister(num_qubits, 'q') 
creg_c = ClassicalRegister(num_qubits, 'c')
circuit = QuantumCircuit(qreg_q, creg_c) 
qubit_array=[]
for i in range(0,input())
#Need four qubits (two inputs, an "add" qubit, and a "carry" qubit)
#qubits initialized in |0> state: cos(0/2)*|0> + sin(0/2)*e^(0i)*|1> = [1 0]^T = |0>
qreg_q = QuantumRegister(4, 'q') 

#Need four classical bits (to store values after measurement)
creg_c = ClassicalRegister(4, 'c') 

#Creates circuit that will act on these qubits and bits


#We want our final state vector to represent the results of a half-adder circuit on 00, 01, 10, 11.
circuit.h(qreg_q[0]) #have a hadamard gate act on q0 --> sets q0 to 0 with 0.5 probability and to 1 with 0.5 probability
circuit.h(qreg_q[1]) #same for q1

#We want s2 to be our "sum" bit. 
#So, whenever q0 is 1, q2 switches from 0 to 1. If q1 is 1, q2 switches back from 1 to 0 (and we carry a bit to q3). 
#If either q0 or q1 is 0, q2 remains unmodified by that qubit. Hence, we get:
# q0 q1 q2 q3
#  0  0  0  0 (q2 unchanged) --> sum = 0 
#  0  1  1  0 (q2 unchanged by q0, flipped by q1) --> sum = 1
#  1  0  1  0 (q2 flipped by q0, unchanged by q1) --> sum = 10
#  1  1  0  1 (q2 flipped by q0, flipped back by q1) --> sum 10
#The carry to q3 happens when q0 and q1 are both 1.
circuit.cx(qreg_q[0], qreg_q[2]) #CNOT gate with control q0 and target q2 (flips q2 if q0 is 1, doesn't change q2 if q0 is 0)
circuit.cx(qreg_q[1], qreg_q[2]) #CNOT gate with control q0 and target q2 (flips q2 if q1 is 1, doesn't change q2 if q1 is 0)
circuit.ccx(qreg_q[0], qreg_q[1], qreg_q[3]) #CCNOT gate, with controls q0 and q1 (flips q3, setting it to 1, iff q0 and q1 are both 1)
