#Develop and apply phase estimator algorithm
def my_qpe(w_qubits,s_qubits, gate, initial_state = None, trotter_number = 1):
    repetitions=1
    #defining a qpe circuit
    qpe_0 = QuantumCircuit(w_qubits+s_qubits,w_qubits)
    if (initial_state != None):
        #initializing the state
        qpe_0.initialize(initial_state,list(range(w_qubits,w_qubits+s_qubits)))
    for i in range(w_qubits):
        qpe_0.h(i)
    #to perform trotterization
    for j in range(trotter_number):
        for counting_qubit in range(w_qubits):
            #to perform U^k operations where k is repetitions
            for i in range(repetitions):
                qubit_list = [counting_qubit]+list(range(w_qubits,w_qubits+s_qubits))
                qpe_0.append(gate,qubit_list)
            repetitions *= 2
        repetitions = 1
    #used inbuilt qft to perform inverse qft and implemented swap
    qpe_1 = QFT(w_qubits, 0, True , True)
    l = [*range(w_qubits)]
    #finally composed qpe0 and inverse qft
    qpe = qpe_0.compose(qpe_1, l)
    return qpe

#Make oracle matrix and diffusion matrix -> create grover operator
def oracle_matrix(indices):
    my_array = np.identity(2 ** 4)
    for i in indices:
      my_array[i, i] = -1
    return my_array

def diffusion_matrix():
    psi_piece = (1 / 2 ** 4) * np.ones(2 ** 4)
    ident_piece = np.identity(2 ** 4)
    return 2 * psi_piece - ident_piece


def grover_operator(indices):
    return np.dot(diffusion_matrix(), oracle_matrix(indices))

def apply_qpe(operator,initial_state)
    qpe = my_qpe(3,1, operator, initial_state = initial_state)
    qpe.measure([0,1,2],[0,1,2])
    result = execute(qpe, backend = simulator, shots = 8192).result()
    count = result.get_counts(qpe)
    display(plot_histogram(count))

#Phase estimator with T-gate operator
initial_state = [1,0]
cir = QuantumCircuit(1)
cir.t(0)
t_gate = cir.to_gate().control(1)
apply_qpe(t_gate,initial_state)

#Phase estimator with Grover operator
apply_qpe(grover_operator[0],initial_state)
