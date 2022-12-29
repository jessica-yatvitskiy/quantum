#My own implementation of a Quantum Circuit, without Qiskit.
#In this implementation, for controlled gates, it is required that
#the control qubits be adjacent to the target qubit.
#If you want to control a qubit that is not adjacent,
#you will have to use repeated swap gates to move it
#so that it is adjacent to your controller bit,
#then apply your gate,
#and then use swap gates again to move the controlled (target) qubit back.

import math
import numpy as np
import random

#Makes a Qubit object, with a number id and a label.
class Qubit(object):
    def __init__(self, arg, label='q'):
        super(Qubit, self).__init__()
        self.arg = arg
        self.label = label

#Makes a Quantum Register to keep track of qubits in circuit
class QuantumRegister(object):
    def __init__(self, num_q, label='qreg'):
        super(QuantumRegister, self).__init__()
        self.size = num_q
        self.label = label
        self.array = [Qubit(i) for i in range(num_q)] #Stores list of all Qubits in circuit
        #Stores the eigenvalues of the state vector representing all the qubits in the circuit
        self.state = np.array([1] + [0] * (2 ** num_q - 1), dtype=complex)
        #For example, if we have 2 qubits in our circuit, our state vector will start out as:
        #1*|00>+0*|01>+0*|10>+0*|11>, and our self.state array will hold the eigenvalues of this vector: 1, 0, 0, 0

#Makes a Classical Register for storing measurements of qubits
class ClassicalRegister(object):
    """ClassicalRegister is where we keep track of measurement outcomes"""
    def __init__(self, num_c, label='creg'):
        super(ClassicalRegister, self).__init__()
        self.size = num_c
        self.label = label
        self.state = np.array([0 for _ in range(num_c)])

#Makes a Gate object
class Gate(object):
    def __init__(self, name, num_q, matrix):
        super(Gate, self).__init__()
        self.name = name
        self.num_q = num_q #number of qubits the gate acts on
        self.matrix = matrix #matrix representation of gate

#Makes a Quantum Circuit object
class QuantumCircuit(object):
    def __init__(self, num_q, num_c):
        super(QuantumCircuit, self).__init__()
        self.num_q = num_q #number of qubits in circuit
        self.qubits = QuantumRegister(num_q) # initialized qubits
        self.num_c = num_c
        self.cbits = ClassicalRegister(num_c) # initialized cbits
        self.circuit = [] # sequence of instructions
        self.pc = 0 # program counter
        self.curr_state = self.qubits.state # state up to the point of program counter

    def _append(self, operation, q_array, c_array):
        # Add new instruction to circuit
        #operation represents the gate we are applying; q_array represents the qubits the gate will be applied to
        #if the gate is a measure gate, the values after measurement will be stored in c_array
        instruction = [operation, q_array, c_array]
        self.circuit.append(instruction)

    def __repr__(self):
        # For displaying quantum circuit
        qasm = ['\n======<QASM>======']
        qasm += ['Qreg: %d, Creg: %d' % (self.num_q, self.num_c)]
        for inst in self.circuit:
            (op, q_arr, c_arr) = inst
            inst_str = '%s ' % op.name
            for q in q_arr:
                qubit = self.qubits.array[q]
                inst_str += '%s%d ' % (qubit.label, qubit.arg)
            inst_str += ', '
            for c in c_arr:
                inst_str += '%s%d ' % (self.cbits.label, c)
            qasm.append(inst_str)
        qasm.append('===============================\n')
        return "\n".join(qasm)

    # When this method is called, a hadamard gate that acts on the given qubit is added to the circuit's instruction list
    def h(self, qubit):
        HGate = Gate('h', 1, 1/np.sqrt(2) * np.array([[1,1],[1,-1]], dtype=complex)) #Creates Gate object representing Hadamard gate
        self._append(HGate, [qubit], [])  #adds to instruction list
        return

    # Pauli X gate
    def x(self, qubit):
        XGate = Gate('x', 1, np.array([[0,1],[1,0]], dtype=complex))
        self._append(XGate, [qubit], [])
        return

    # Pauli Y gate
    def y(self, qubit):
        YGate = Gate('y', 1, np.array([[0,-1*j],[1*j,0]], dtype=complex))
        self._append(YGate, [qubit], [])
        return

    # Pauli Z gate
    def z(self, qubit):
        ZGate = Gate('z', 1, np.array([[1,0],[0,-1]], dtype=complex))
        self._append(ZGate, [qubit], [])
        return

    # Phase gate (sqrt(Z))
    def s(self, qubit):
        SGate = Gate('s', 1, np.array([[1,0],[0,1*j]], dtype=complex))
        self._append(SGate, [qubit], [])
        return

    # S dagger gate
    def sdg(self, qubit):
        SGate = Gate('sdg', 1, np.array([[1,0],[0,-1*j]], dtype=complex))
        self._append(SGate, [qubit], [])
        return

    # T gate
    def t(self, qubit):
        TGate = Gate('t', 1, np.array([[1,0],[0,complex(1/math.sqrt(2),1/math.sqrt(2))]]), dtype=complex)
        self._append(TGate, [qubit], [])
        return

    # T dagger gate
    def tdg(self, qubit):
        TGate = Gate('tdg', 1, np.array([[1,0],[0,complex(1/math.sqrt(2),-1/math.sqrt(2))]]), dtype=complex)
        self._append(TGate, [qubit], [])
        return

    # Controlled X gate (CNOT)
    def cx(self, ctrl_qubit, trgt_qubit):
        CXGate = Gate('cx', 2, np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex))
        self._append(CXGate, [ctrl_qubit, trgt_qubit], [])
        return

    # Controlled Z gate (CZ)
    def cz(self, ctrl_qubit, trgt_qubit):
        CZGate = Gate('cz', 2, np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,-1]], dtype=complex))
        self._append(CZGate, [ctrl_qubit,trgt_qubit], [])
        return

    # Toffoli gate
    def toffoli(self, ctrl_qubit_1, ctrl_qubit_2, trgt_qubit):
        TFGate = Gate('tf', 3, np.array([[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,1],[0,0,0,0,0,0,1,0]], dtype=complex))
        self._append(TFGate, [ctrl_qubit_1,ctrl_qubit_2,trgt_qubit], [])
        return

    #Swap gate
    def swap(self, qubit_1, qubit_2):
        SWGate = Gate('sw', 2, np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=complex))
        self._append(SWGate, [qubit_1, qubit_2], [])
        return

    # Creates Measurement gate
    # Note: The actual measurement is done in evolveOneStep
    def measure(self, qubits, cbits):
        assert(len(qubits) == len(cbits))
        Measure = Gate('measure', len(qubits), None)
        self._append(Measure, qubits, cbits)
        return

    #When we apply a gate to a some qubits (represented by q_arr), we want the circuit's overall state vector to be updated accordingly
    #Here, we convert the given gate's matrix into a form such that, when later multiplied by the circuit's current state vector,
    #it will achieve this effect.
    #Say we have n qubits and we want to apply our gate to qubits i and j. Then,
    #We return the tensor product of i identity matrices, the gate matrix, and n-j identity matrices.
    #This is equivalent to the tensor product of an identity matrix of dimensions i*i, the gate matrix,
    #and an identity matrix of dimensions (n-j)*(n-j).
    def tensorizeGate(self, gate, q_arr):
        q_arr=np.array(q_arr)
        if len(q_arr)>1:
            try:
                assert(q_arr[1:len(q_arr)]==q_arr[0:len(q_arr)-1]+1) #Make sure we are only using adjacent qubits
            except AssertionError as msg:
                print("Qubits for gate must be adjacent. If you want to use non-adjacent qubits, use the swap gate to move them together.")
                print("Then apply your gate, and use the swap gate to move them back apart again.")
                assert(q_arr[1:len(q_arr)]==q_arr[0:len(q_arr)-1]+1)
        q_id1=q_arr[0] #start of qubits that the gate will be applied to
        q_id2=q_arr[-1] #end of qubits that the gate will be applied to
        Mat1=np.identity(2**q_id1) #create first identity matrix
        Mat2=gate.matrix
        Mat3=np.identity(2**(self.num_q-q_id2-1)) #create second identity matrix
        Prod1=np.kron(Mat1,Mat2)
        return(np.kron(Prod1,Mat3)) #return the tensor product

    #Evaluates the current instruction (the program counter, self.pc stores the index of the curr instruction in the
    #circuit's instruction list)
    #Updates the state vector accordingly
    def evolveOneStep(self):
        curr_state = self.curr_state
        #from current instruction, extract the operation (gate), and the qubits (and possibly classical bits) the gate will be applied to
        (op, q_arr, c_arr) = self.circuit[self.pc]
        if op.name != 'measure': #if the gate is not a measure gate, we just update the state vector according to the instruction
            unitary = self.tensorizeGate(op, q_arr) #converts the gate into a form such that its action will be reflected in the state vector when multiplied
            curr_state = unitary @ curr_state #multiply tensorized gate and state vector to get updated state vector
            self.curr_state = curr_state.reshape((2**self.num_q))
            self.pc += 1 #increase program counter, so that it points to the next instruction
            return False #returns False to represent that we have not performed measurement yet
        else:
            print("Already reached end of circuit (excluding measurements).")
            #Generate probability distribution for qubit states from state vector
            probabilities = abs(self.curr_state)**2
            #Based on the probability distribution, pick a state
            result=bin(random.choices(list(range(0,len(self.curr_state))), weights=probabilities,k=1)[0])
            #Convert to binary, to get the values of each qubit in that state
            res = [int(x) for x in str(result[2:])]
            res=np.array([0]*(self.num_q-len(res))+res)
            #Store the values of the qubits that we wanted to measure in the correspondin classical bits
            self.cbits.state[c_arr]=res[q_arr]
            self.pc+=1 #increase program counter, so that it points to the next instruction
            return True #returns True to represent that we have performed measurement

    def simulate(self):
        # A function for simulating circuit from scratch (from pc = 0).
        measured = False
        num_instr=len(self.circuit)
        while self.pc<num_instr: #keeps going through circuit's instruction list and updating state vector accordingly, until we are done with instructions
            measured=self.evolveOneStep()
        self.qubits.state = self.curr_state
        #if we measured at the end of our run through the circuit, we output the result of the measurement
        if measured:
            return self.cbits.state
        #otherwise, we return the final states of all our qubits
        else:
            return self.qubits.state

def testSimulate():
    qc = QuantumCircuit(2,2)
    qc.h(0)
    qc.swap(0,1)
    qc.measure([0,1],[0,1])
    outcome = qc.simulate()
    print(outcome)

testSimulate()
testSimulate()
testSimulate()
