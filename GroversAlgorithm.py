#Grover's algorithm
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from numpy import pi

qreg_q = QuantumRegister(2, 'q')
creg_c = ClassicalRegister(2, 'c')
circuit = QuantumCircuit(qreg_q, creg_c)

#In the first part of this circuit, we're the Password Keeper.
#We want our password to be the two-bit password 11, represented by the two-qubit state |11>.
#So, we build an Oracle to "mark" the "11" state as the "password", by changing the
#circuit's state vector in such a way that the change cannot be perceived through a simple measurement
#(so that it's not too easy for someone to find the password).
#So, initially, our circuit's state vector is 1|00>+0|01>+0|10>+0|11>, because our two qubits are initially set to 0.
#Then, we apply Hadamard gates to put the states of our qubits into equal superposition, so that our state vector becomes
#0.5|00>+0.5|01>+0.5|10>+0.5|11>, because the probability of each state is now 1/4 or (0.5)^2.
#Finally, we apply the controlled-Z gate, which negates the value of the second qubit's |1> state when the first qubit is in the |1> state,
#setting our state vector to 0.5|00>+0.5|01>+0.5|10>-0.5|11>. In this way, the |11> state is differentiated from the other states in such a
#way that cannot be detected through measurement, but can be detected through Grover's algorithm.
#The job of Grover's algorithm is to take the output of the Oracle (the state vector after being "marked")
#and manipulate it so that the "marking" becomes something that we can measure, and therefore find the correct password.
circuit.h(qreg_q[0])
circuit.h(qreg_q[1])
circuit.cz(qreg_q[1], qreg_q[0])

#Applying Grover's:
#It's been proven that, for an N bit "password", sqrt(N) steps of Grover's algorithm are sufficient to
#convert our state vector into a form that, when measured, is highly likely to evaluate to the
#"password", i.e. the secret N-qubit state we are looking for. Here we have two bits, so
#1 step should be about enough.
#I'm not going to go over the logic behind Grover's here, but this article from IBM Quantum has a
#great explanation: https://quantum-computing.ibm.com/composer/docs/iqx/guide/grovers-algorithm
#But, basically, we take the state vector that was the result of the Oracle, which has the correct "password" marked subtly.
#Remember, it's marked in such a way that simply measuring our current state won't reveal the "password".
#So, instead, we take the state vector given to us by the Oracle, and "amplify" this "marking", this difference from the other qubits.
#We do this "reflecting" all states perpendicular to our current state vector across our state vector.
#(Again, the reasoning for this is well explained in the article above).
#An easy way to do this reflection is to:
#1) Convert our state vector to an equal superposition of states, using Hadamard gates; call this new state S'.
#2) Reflect all perpendicular vectors to S' over S'. This is done by negating all sub-states (01, 10, 11) other than 00.
#3) Convert back to our state vector, while maintaining this negation.
#This is all done by our Z, Z, CZ gates below.
#Now, our state vector is in the desired form, such that measuring will now give us the correct "password"
#with high probability.
circuit.h(qreg_q[0])
circuit.h(qreg_q[1])
circuit.z(qreg_q[0])
circuit.z(qreg_q[1])
circuit.cz(qreg_q[1], qreg_q[0])
circuit.h(qreg_q[0])
circuit.measure(qreg_q[0], creg_c[0])
circuit.measure(qreg_q[1], creg_c[1])
