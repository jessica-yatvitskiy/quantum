import numpy as np
import copy

#Problem to be addressed:
#A common limitation of real devices is that not all connections between qubits
#are allowed. For example, for certain hardware structures (or topologies), we
#cannot apply a CNOT gate between qubits i and j.
#However, we can address this problem by using by swapping the target qubit j
#with a qubit that qubit i can control, applying the CNOT gate, and swapping qubit j back to its original position.
#The purpose of this program is to find the shortest list of swaps that must be conducted in order to achieve the CNOT connection we want.

#We have a file graph.txt that stores, for each qubit, the qubits that it has connections with.
#This function outputs, in matrix form, this data, so that we can use it in our next function to find the shortest sequence of swaps
#to determine the CNOT connection.
def get_graph(file_name):
    f = open(file_name, "r")
    lines = f.readlines()
    graph=[]
    for line in lines:
        curr_node_neighbors=[]
        i=0
        while i<len(line) and i!="\n":
            curr_node_neighbors.append(int(line[i]))
            i+=3
        graph.append(curr_node_neighbors)
    return(graph)

#This function takes the control qubit, the target qubit, the matrix of qubit connections, and outputs
#the minimum number of swaps necessary to achieve our desired connection between the control and target qubits.
#Meanwhile, it also fills up the path_list array, which contains, in reverse order, the qubits we need to swap our
#target qubit with in order to achieve the desired connection with the control qubit.
path_list=[]
def min_path_dist(curr_node,end_node,graph,curr_len,max_iter):
    if curr_len>=max_iter:
        return(max_iter)
    if end_node in graph[curr_node]:
        return(0)
    else:
        min_path_len=max_iter
        for i in range(0,len(graph[curr_node])):
            neighbor=graph[curr_node][i]
            curr_path_len=1+min_path_dist(neighbor,end_node,graph,curr_len+1,min_path_len)
            if curr_path_len<min_path_len:
                min_path_len=curr_path_len
                path_list.append(graph[curr_node][i])
        return(min_path_len)
#Print minimum total number of swaps necessary (forward and back)
min_num_swaps=min_path_dist(0,9,get_graph("graph.txt"),0,5)*2
print("minimum total swaps necessary:")
print(min_num_swaps)
#Print list of qubits we need to swap our target qubit with (before we apply the gate)
path_back=copy.deepcopy(path_list)
path_list.reverse()
print("swaps forward:")
print(path_list)
#Print list of qubits we need to swap our target qubit with (to move it back to its original position after applying the gate)
print("and swaps back:")
print(path_back)
