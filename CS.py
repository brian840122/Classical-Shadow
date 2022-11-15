import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
import os

if not os.path.exists(f"./results"):
    os.makedirs(f"./results")
if not os.path.exists(f"./images"):
    os.makedirs(f"./images")

# create a table cor_states to tell our machine what the corresponding density matrix is after we get the measurement outcome
x_plus = np.array([[1, 1], [1, 1]], dtype=complex, requires_grad=False)*3/2-[[1, 0], [0, 1]]
x_minus = np.array([[1, -1], [-1, 1]], dtype=complex, requires_grad=False)*3/2-[[1, 0], [0, 1]]
y_plus = np.array([[1, -1j], [1j, 1]], dtype=complex, requires_grad=False)*3/2-[[1, 0], [0, 1]]
y_minus = np.array([[1, 1j], [-1j, 1]], dtype=complex, requires_grad=False)*3/2-[[1, 0], [0, 1]]
z_plus = np.array([[1, 0], [0, 0]], dtype=complex, requires_grad=False)*3-[[1, 0], [0, 1]]
z_minus = np.array([[0, 0], [0, 1]], dtype=complex, requires_grad=False)*3-[[1, 0], [0, 1]]
cor_states = [[x_plus, x_minus], [y_plus, y_minus], [z_plus, z_minus]]

n_try = 10 # number of this experiments, the final results average over all experiments
T=100000 # number of measurements, effects the performance of predicting our target density matrix
meas = [qml.PauliX, qml.PauliY, qml.PauliZ]

max_nq = 4
for nq in range(max_nq):
    n_qubit = nq+1
    dev = qml.device('default.qubit', wires=n_qubit, shots=1)

    @qml.qnode(dev)
    def circuit(params, meas_):
        #prepare our target state
        p = np.array_split(params, max_nq*2)
        for i in range(n_qubit):
            qml.Rot(*p[i], wires=i)
        if n_qubit>1:
            for i in range(n_qubit-1):
                qml.CNOT(wires=[i, i+1])
        for i in range(n_qubit):
            qml.Rot(*p[i+max_nq], wires=i)
        if n_qubit>1:
            for i in range(n_qubit-1):
                qml.CNOT(wires=[i, i+1])
        return [qml.sample(meas[meas_[i]](wires=i)) for i in range(n_qubit)]

    @qml.qnode(dev)
    def target_dm(params):
        #prepare our target state
        p = np.array_split(params, max_nq*2)
        for i in range(n_qubit):
            qml.Rot(*p[i], wires=i)
        if n_qubit>1:
            for i in range(n_qubit-1):
                qml.CNOT(wires=[i, i+1])
        for i in range(n_qubit):
            qml.Rot(*p[i+max_nq], wires=i)
        if n_qubit>1:
            for i in range(n_qubit-1):
                qml.CNOT(wires=[i, i+1])
        return qml.density_matrix([i for i in range(n_qubit)])



    trace_dists = np.zeros(T//200)
    for r in range(n_try):
        params = np.random.randn(max_nq*6)
        target = target_dm(params)
        dm = np.zeros((2**n_qubit, 2**n_qubit), dtype=complex) 
        trace_dist = []
        for i in range(T):
            meas_ = np.random.choice(3, n_qubit)
            m_results = 1/2-circuit(params, meas_)/2
            m_results = [int(m_results[i]) for i in range(len(m_results))]
            dm_ = [1]
            for n in range(n_qubit):
                dm_ = np.kron(dm_, cor_states[meas_[n]][m_results[n]])
            #dms.append(dm)
            dm += dm_
            if not (i+1)%200:
                dm_i = dm/i
                D = target - dm_i
                trace_dist.append(np.real(np.sqrt(np.trace(D.conjugate().transpose() @ D))))
        trace_dists += trace_dist
    trace_dists = trace_dists/n_try
    x = [200*(i+1) for i in range(T//200)]
    y = trace_dists
    plt.plot(x, y, label=f'{n_qubit}_qubit(s)')
    f = open(f"./results/trace_distance_{nq}_qubits.txt",'w+')
    f.write(str(y))
    f.close()

plt.ylabel("Trace Distance")
plt.xlabel("Number of measurements")
plt.legend()
plt.savefig('./images/image4')
plt.show()
#D = target - dm
#print(np.sqrt(np.trace(D.conjugate().transpose() @ D)))
