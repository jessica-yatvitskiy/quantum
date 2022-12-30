#Implements a full adder circuit for any number of bits; builds on half adder

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from numpy import pi

#Gets user input for length in bits (qubits in |0> or |1> state) of sequences to add
#For example, if the user inputs "2", we create a circuit that computes the sum of
#all possible pairs of 2-bit sequences, i.e. 00+00, 00+01, 00+10, 00+11, ... , 11+00, 11+01, 11+10, 11+11
print("Please input a number representing the length in 'bits' (qubits that are set to 0 or 1) of each number we will be adding:")
length=int(input())

#Initialize circuit
qreg_q = QuantumRegister(6*length+1, 'q') #Create quantum register with the number of qubits we'll need
creg_c = ClassicalRegister(6*length+1, 'c') #Classical register
circuit = QuantumCircuit(qreg_q, creg_c) #Initialize circuit

#Apply hadamard gate to each of the input qubits
#For example, if the user-input length is 4, qubits 0-3, representing the first qubit sequence, are
#set to 1/sqrt(2)*|0> + 1/sqrt(2)*|1> = [1/sqrt(2) 1/sqrt(2)]^T,
#so that they have equal probability to be in state |0> or state |1>.
#The same is done for qubits 4-7, representing the second qubit sequence.
for i in range(0,2*length):
    circuit.h(qreg_q[i])

#Continue building circuit
#Imagine these are our sequences:
# 0 0 1 1
# 1 0 1 0
#We go through each column, right to left.
#In each step (for each column), we compute a sum bit using the two bits in the column and a carry bit
#(The carry bit is set to 0 for the first column.)
#We also compute a new carry bit, to be applied to the next column.
#So, for our example, we set the sum bit for column 1 to 1, by applying a CNOT gate to the sum bit with the control bits
#being the two bits in column 1 and the current carry bit (0).
#We set the new carry bit, to be applied in the next column, to 0.
#We do this by using CNOT, CCNOT, and NOT gates
#to basically express (bit 1 AND bit 2) or [(bit1 XOR bit2) AND (old carry bit)],
#meaning we only carry if any pair of bits from the set (bit1, bit2, carry bit) are both 1.
#We repeat this for each of the following columns:
#compute sum bit using bit1, bit2, and current carry bit, in pretty much the same way the half-adder does it
#compute new carry bit, by setting it to 1 only if 2 of the bits from [bit1, bit2, old-carry-bit] are 1.
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

#Draw the circuit and save it to a file
circuit.draw(output='mpl',filename='full_adder.png')
