#Here, I've written methods to implement various complex gates, which could be used as Oracles.

import math
import numpy as np

#And gate
#Iff both input qubits are 1, the output qubit is set to 1
def AND(qc, in_qubit_1, in_qubit_2, out_qubit):
    qc.ccx(in_qubit_1, in_qubit_2, out_qubit)
    return

#Or gate
#Iff both input qubits are 0, the output qubits is set to 0.
def OR(qc, in_qubit_1, in_qubit_2, out_qubit):
    qc.x(in_qubit_1)
    qc.x(in_qubit_2)
    qc.ccx(out_qubit)
    qc.x(out_qubit)
    return

#N-bit Nand gate
#Iff some input qubit(s) is/are 0, the output qubit is set to 1.
def nAND(qc, in_qubits, out_qubit, ancilla):
    qc.ccx(in_qubits[0],in_qubits[1],ancilla[0])
    for i in range(0,len(in_qubits)-2):
        #If there is an input qubit that is 0, an ancilla qubit will be set to 0, so all further ancilla qubits will be set to 0.
        qc.ccx(ancilla[i], in_qubits[i+2], ancilla[i+1])
    #The output qubit is set to become the opposite of the last ancilla qubit.
    qc.cx(ancilla[len(in_qubits)-1],out_qubit)
    qc.x(out_qubit)
    return

#N-bit Nand gate
#Iff all input qubits are 0, the output qubit is set to 1.
def nOR(qc, in_qubits, out_qubit, ancilla):
    qc.x(in_qubits[0])
    qc.x(in_qubits[1])
    qc.ccx(in_qubits[0],in_qubits[1],ancilla[0])
    for i in range(0,len(in_qubits)-2):
        qc.x(in_qubits[i+2])
        #If there is an input qubit that is 1, an ancilla qubit will be set to 0, so all further ancilla qubits will be set to 0.
        qc.ccx(ancilla[i], in_qubits[i+2], ancilla[i+1])
    #The output qubit is set to become the same as the last ancilla qubit.
    qc.cx(ancilla[len(in_qubits)-1],out_qubit)
    return

#3-bit Majority gate
#Iff at least two input qubits are 0, the output qubit is set to 0.
#Otherwise, the output qubit is set to 1.
def MAJ(qc, in_qubit_1, in_qubit_2, in_qubit_3, out_qubit):
    #Only flips the last two qubits if the first qubit is 1.
    qc.cx(in_qubit_1,in_qubit_2)
    qc.cx(in_qubit_1,in_qubit_3)
    #Only flips the first qubit if the last two qubits are 1. This sets the first qubit to the majority value.
    #If the last two qubits are 1 because they were flipped, this means the first qubit was 1, while the others were 0 --> q0 = 0 (flipped)
    #If the last two qubits are 1 because they were NOT flipped, this means the first qubit was 0, while the others were 1 --> q0 = 1 (flipped)
    #If the last two qubits are 0 because they were flipped, this means the first qubit was 1, and the others were also 1 --> q0 = 1 (unflipped)
    #If the last two qubits are 0 because they were NOT flipped, this means the first qubit was 0, and the others were also 0 --> q0 = 0 (unflipped)
    #If the last two qubits are different from each other, this means they were originally such, so the first qubit is in the majority --> q0 = old q0. (unflipped)
    qc.ccx(in_qubit_2,in_qubit_3,in_qubit1)
    #Since the first qubit now reflects the majority value, we set the output qubit to become the same as the first qubit.
    qc.cx(in_qubit_1,out_qubit)
    return

#Reflects the state vector over the |0...> state
def ReflectZero(qc, qubits, ancilla):
    #Using the same logic as the nOr function, the last ancilla qubit is set to 1 if we are in the |0...> state, and 0 otheriwse
    qc.x(qubits[0])
    qc.x(qubits[1])
    qc.ccx(qubits[0],qubits[1],ancilla[0])
    for i in range(0,len(qubits)-2):
        qc.x(qubits[i+2])
        qc.ccx(ancilla[i], qubits[i+2], ancilla[i+1])
    #If we are in the |0..> state, we flip our sign to zero by applying the controlled-z gate.
    qc.cx(ancilla[len(qubits)-1], qubits[0])
    qc.cz(ancilla[len(qubits)-1], qubits[0])
    qc.cx(ancilla[len(qubits)-1], qubits[0])
    return

#Reflects the state vector over the |+...> state.
#Applies Hadamard gate to turn all |+> qubits into |0> qubits and all |-> qubits into |1> Qubits.
#Applies same logic as ReflectZero.
#Reapplies Hadamard gate to each qubit to turn them back into original |+/-> states.
def ReflectUniform(qc, qubits, ancilla):
    qc.h(qubits[0])
    qc.x(qubits[0])
    qc.h(qubits[1])
    qc.x(qubits[1])
    qc.ccx(qubits[0],qubits[1],ancilla[0])
    for i in range(0,len(qubits)-1):
        qc.h(qubits[i])
        qc.x(qubits[i+2])
        qc.ccx(ancilla[i], qubits[i+2], ancilla[i+1])
    for i in range(0,len(qubits)):
        qc.h(qubits[i])
    qc.cx(ancilla[len(qubits)-1], qubits[0])
    qc.cz(ancilla[len(qubits)-1], qubits[0])
    qc.cx(ancilla[len(qubits)-1], qubits[0])
    return
