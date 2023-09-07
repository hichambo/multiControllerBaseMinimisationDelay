import sys
import math
import numpy as np
from sklearn.cluster import KMeans
import networkx as nx
import matplotlib.pyplot as plt
import random

N = 30
def binary_identifier(N):
  alpha = np.zeros((N, N), dtype=int)
  for i in range(N):
    for j in range( N):
        if i==j:
             alpha[i][j]=1
  for i in range(N):
    for j in range(i+1, N):
        alpha[i][j] = random.randint(0, 1)
        alpha[j][i] = alpha[i][j]
  return alpha

def calculate_delay_matrix(alpha):
    
    N = len(alpha)
    delay = np.zeros((N, N),dtype=int)

    for i in range(N):
        for j in range(N):
            if i==j:
                delay[i][j]=0
            elif alpha[i][j] == 1:
                # Calculate the delay between switches i and j
                delay[i][j] =random.randint(1, 9)# Replace with your own delay calculation logic
                delay[j][i]=delay[i][j]

    return delay

alpha=binary_identifier(N)
delay = calculate_delay_matrix(alpha)
# Generate the binary matrix
binary_matrix = binary_identifier(N)
# Save the binary matrix to a text file
np.savetxt('Binary_Identifier.txt', binary_matrix, fmt='%d')

# Generate the DELAY matrix
# Save the binary matrix to a text file
np.savetxt('ArrayD.txt', delay, fmt='%d')

def optimal_strategy(N, Binary_Identifier, D):
    # Initialize a graph object
    graph = Graph(N)
    # Initialize a matrix to store the minimum delays
    D_prime = [[0 for j in range(N)] for i in range(N)]
    # Construct the graph
    for i in range(N):
        for j in range(N):
            if Binary_Identifier[i][j] == 1:
                graph.graph[i][j] = D[i][j]
    # Apply Dijkstra's algorithm to each pair of non-adjacent switches
    for i in range(N):
        for j in range(N):
            if Binary_Identifier[i][j] != 1:
                # Reset the graph object
                graph = Graph(N)
                # Construct the graph
                for k in range(N):
                    for l in range(N):
                        if Binary_Identifier[k][l] == 1:
                            graph.graph[k][l] = D[k][l]
                # Apply Dijkstra's algorithm to find the shortest path between i and j
                graph.dijkstra(i)
                D_prime[i][j] = graph.dist[j]
    return D_prime

#  This function is to concatenate to matrix the first is delay between the direct switch and the second is the nonadjacent switch 
def total_matrix_delay(matrix1, matrix2):#input to matrix witch you want to concatenate
    final_matrix = [] #initialization to zero the matrix will be returned
    for i in range(len(matrix1)): 
        row = []
        for j in range(len(matrix1[0])):
            if matrix1[i][j] == 0:
                row.append(matrix2[i][j])
            else:
                row.append(matrix1[i][j])
        final_matrix.append(row)
    return final_matrix

# this function is to determin the sum of each row in vector
def sum_delay(matrix):
    row_sums = []
    for row in matrix:
        row_sum = sum(row)
        row_sums.append(row_sum)
    return row_sums

# This function is to determine the index and value of the minimum of vector
def min_index_delay(vec):
    min_val = float('inf')
    min_idx = -1
    
    for i, val in enumerate(vec):
        if val < min_val:
            min_val = val
            min_idx = i
            
    return min_idx, min_val

# This function is to update the position of value you want to set a infini 
def update_value_to_infinity(vector, position):
    vector[position] = math.inf
    return vector

def Association(total_matrix,Yi): 
  #the binary association identifier between the ith controller and the jth switch
  #initialization of Xij to zero
  binary_association = [[0] * N for _ in range(N)]
  Controller_val=[]
  # assocaite the to the each switch with controller offer the min value
 
  for i in range(N):
        Xij = [0] * N # inisialise vector with zero 
        for j in range(N):
            if Yi[j] == 1:# test if the current case is controller 
                Xij[j]= total_matrix[i][j] # insert each value of controller
            else:
                Xij[j]=math.inf #insert each value is not controller to infini to not selected 
        # select the index of minimum value of controller offre 
        inx,val=min_index_delay(Xij)
        Controller_val.append(np.sum(val))
        # charge the value selected in matrix of 
        for j in range(N):
             if j==inx and i != j:# check the index of j if is the index of minimun switch slected and if ii != j to ensures that a switch is not associated with itself as a controlle
                binary_association[i][inx] = 1 # associate the switch i to controller  
  return binary_association,Controller_val

# inotialize the graph
class Graph():
    def __init__(self, vertices):
        self.V = vertices
        self.graph = [[0 for column in range(vertices)] for row in range(vertices)]
        self.dist = [sys.maxsize] * vertices

    def printSolution(self, dist):
        print("Vertex Distance from Source")
        for node in range(self.V):
            print(node, "\t", dist[node])

    def minDistance(self, dist, sptSet):
        min = sys.maxsize
        for v in range(self.V):
            if dist[v] < min and sptSet[v] == False:
                min = dist[v]
                min_index = v
        return min_index

    def dijkstra(self, src):
        self.dist = [sys.maxsize] * self.V
        self.dist[src] = 0
        sptSet = [False] * self.V

        for cout in range(self.V):
            u = self.minDistance(self.dist, sptSet)
            sptSet[u] = True
            for v in range(self.V):
                if self.graph[u][v] > 0 and sptSet[v] == False and self.dist[v] > self.dist[u] + self.graph[u][v]:
                    self.dist[v] = self.dist[u] + self.graph[u][v]
       
       
M=7 #the number of controller in your network topology
#switches = ['s1', 's2', 's3', 's4', 's5','s6','s7', 's8', 's9','s10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21', 's22', 's23', 's24', 's25', 's26', 's27', 's28', 's29', 's30'] #set of switch in your network topology
# Read Binary_Identifier matrix from file
array_file = 'Binary_Identifier.txt'
Binary_Identifier = []
with open(array_file, 'r') as file:
    for line in file:
        row = list(map(int, line.strip().split()))
        Binary_Identifier.append(row)

# Read D matrix from file
D_file = 'ArrayD.txt'
D = []
with open(D_file, 'r') as file:
    for line in file:
        row = list(map(int, line.strip().split()))
        D.append(row)
# controller= [0]* M


# k-means controller placement 
#Determine the first controller:
D_prime= optimal_strategy(N, Binary_Identifier, D)# the delay between any two adjacent switches in the network 
total_matrix=total_matrix_delay(D,D_prime) # The total matrix with the delay in adjacent and noadjacent swithch 
Sum_Delay=sum_delay(total_matrix_delay(D,D_prime))# the sum of minimum delay
#Convert the sum_delay vector to a string representation to Print the modified vector with the colored values
sum_delay_str = ', '.join(str(element) for element in Sum_Delay)

index_minimum_delay,min_delay=min_index_delay(Sum_Delay)# Index and the sum of the minimum delay with other switches
delay_moyen=[]
delay_moyen.append(min_delay)
Yi = [0] * N
Yi[index_minimum_delay] = 1 # selact the first controller 

# set the reste of controller  
m=1
# the vecter of witch select controller with witch value selected updates to infin to ll be not select in next step
D_controller= update_value_to_infinity(Sum_Delay, index_minimum_delay)
# repeat the same step to determin all controller corespendate 
while m<M:       
              index_minimum_delay,min_delay=min_index_delay(D_controller)

              Yi[index_minimum_delay]=1
              binary_association,val=Association(total_matrix,Yi)
              delay_moyen.append(np.sum(val))
              D_controller= update_value_to_infinity(Sum_Delay, index_minimum_delay)
              m += 1
delay_moyen_str = ', '.join(str(element) for element in delay_moyen)


# Define the ANSI escape sequence for red and bleu and reset color to change the print color
red_color_sequence = '\033[91m'
reset_color_sequence = '\033[0m'
bleu_color_sequenece = '\033[34m'
#Convert the Yi and D_controller vector to a string representation to Print the modified vector with the colored values
Yi_str = ', '.join(red_color_sequence + str(element) + reset_color_sequence if element == 1 else bleu_color_sequenece + str(element) + reset_color_sequence for element in Yi)
D_controller_str = ', '.join(red_color_sequence + str(element) + reset_color_sequence if element == float('inf') else bleu_color_sequenece + str(element) + reset_color_sequence for element in D_controller)

# Create a NetworkX graph based on the Binary_Identifier       
G_1 = nx.Graph() #initializes an empty graph object called G_1
for i in range(len(Binary_Identifier)): 
    for j in range(i+1, len(Binary_Identifier)):
        if Binary_Identifier[i][j] > 0:
            G_1.add_edge(i+1, j+1)
# Draw the graph without weights and before add controller
pos = nx.spring_layout(G_1)
color_map = ['lightblue'] * len(G_1.nodes)
nx.draw(G_1, pos, with_labels=True, node_color=color_map, node_size=800)
plt.show()


# Create a NetworkX graph object from the total_matrix
G = nx.Graph()
for i in range(len(total_matrix)):
    for j in range(i+1, len(total_matrix)):
        if total_matrix[i][j] > 0:
            G.add_edge(i+1, j+1, weight=total_matrix[i][j])
# Draw the graph with weights
pos = nx.spring_layout(G)
color_map = []
for j in range(N):
    if (Yi[j]==1):
        color_map.append('red')
    else: 
        color_map.append('lightblue') 

nx.draw(G, pos, with_labels=True, node_color=color_map, node_size=800)
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.show()
#------------------------------------------------------------------------------------
# Create a NetworkX graph object from the association matrix to draw the cluster
G_association = nx.Graph()
for i in range(len(binary_association)):
    for j in range(len(binary_association[i])):
        if binary_association[i][j] == 1:
            G_association.add_edge('S:'+str(i+1), 'C: '+str(j+1))
# Draw the clusters
pos_association = nx.spring_layout(G_association)
color_map_association = []
for node in G_association.nodes:
    if 'S:' in node:
        color_map_association.append('lightblue')
    else:
        color_map_association.append('red')
nx.draw(G_association, pos_association, with_labels=True, node_color=color_map_association, node_size=800)
labels_association = nx.get_edge_attributes(G_association, 'weight')
nx.draw_networkx_edge_labels(G_association, pos_association, edge_labels=labels_association)
# Display the plot
plt.show()


print("-----------------------------------------------------")
print("The vector that represents the sum is:",
      bleu_color_sequenece +"["+ sum_delay_str +"]"+reset_color_sequence)
print("-----------------------------------------------------")
print("controller poition:",bleu_color_sequenece +"["+ Yi_str +"]"+reset_color_sequence)
print("-----------------------------------------------------")



print("Delay value after adding each controller:",bleu_color_sequenece +"["+ delay_moyen_str +"]"+reset_color_sequence)


# Sample data
x = list(range(1, M+1))
y = delay_moyen
# Plotting the graph
plt.plot(x, y)
# Adding labels and title
plt.xlabel('Nombre of controllers ')
plt.ylabel('Total control plane delay (ms) ')
plt.title("Delay minimization for controller problem placement")
plt.xticks(range(1, len(x) + 1)) # Set the x-axis ticks by the number of controllers
# Display the graph
plt.show()

