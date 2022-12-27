#Implements a full_adder circuit

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from numpy import pi
from qiskit.quantum_info import Statevector
from qiskit import Aer, execute

print("Please input a number representing the length in 'bits' (qubits that are set to 0 or 1) of each number we will be adding:")
length=int(input())
qreg_q = QuantumRegister(6*length+1, 'q')
creg_c = ClassicalRegister(6*length+1, 'c')
circuit = QuantumCircuit(qreg_q, creg_c)
qubit_array=[]
for i in range(0,2*length):
    circuit.h(qreg_q[i])
for i in range(0,length):
    circuit.cx(qreg_q[i], qreg_q[i+(5*length)])
    circuit.cx(qreg_q[i+length], qreg_q[i+(5*length)])
    circuit.ccx(qreg_q[i+(5*length)], qreg_q[(3*i)+(2*length)],qreg_q[(3*i)+(2*length)+2])
    circuit.x(qreg_q[(3*i)+(2*length)+2])
    circuit.cx(qreg_q[(3*i)+(2*length)], qreg_q[i+(5*length)])
    circuit.ccx(qreg_q[i], qreg_q[i+length],qreg_q[(3*i)+(2*length)+1])
    circuit.x(qreg_q[(3*i)+(2*length)+1])
    if i<(length-1):
        circuit.ccx(qreg_q[(3*i)+(2*length)+1], qreg_q[(3*i)+(2*length)+2],qreg_q[(3*i)+(2*length)+3])
    else:
        circuit.ccx(qreg_q[(3*i)+(2*length)+1], qreg_q[(3*i)+(2*length)+2],qreg_q[6*length])
circuit.draw(output='mpl',filename='full_adder.png')
#result = execute(circuit, backend=backend, shots=1).result()
#print('State Vector:', result.get_statevector() )
