import time
import numpy as np
from scipy.io import loadmat
from collections import deque
import bqphy.BQPhy_Optimiser as qea


# =========================================================
# ORIGINAL HELPERS
# =========================================================
def OmegamatrixFn_CL(orgmainmatrix, FlipedGOmega, M_org, N_org, rowi_sigma_i, colj_sigma_i):
    i = rowi_sigma_i
    j = colj_sigma_i

    mainmatrix = np.array(orgmainmatrix, dtype=float, copy=True)
    FlipedGOmega = np.asarray(FlipedGOmega, dtype=float).reshape(-1)

    M = M_org
    N = N_org

    ivec = list(range(i + 1, M + 1))
    jvec = list(range(i - 1, 0, -1))
    kvec = list(range(j - 1, 0, -1))
    lvec = list(range(j + 1, N + 1))

    maxel = max(len(ivec), len(jvec), len(kvec), len(lvec))

    if len(ivec) < maxel:
        ivec.extend([0] * (maxel - len(ivec)))
    if len(jvec) < maxel:
        jvec.extend([0] * (maxel - len(jvec)))
    if len(kvec) < maxel:
        kvec.extend([0] * (maxel - len(kvec)))
    if len(lvec) < maxel:
        lvec.extend([0] * (maxel - len(lvec)))

    for x in range(maxel):
        if ivec[x] > 0:
            mainmatrix[ivec[x] - 1, :] = FlipedGOmega[x]
        if jvec[x] > 0:
            mainmatrix[jvec[x] - 1, :] = FlipedGOmega[x]
        if kvec[x] > 0:
            mainmatrix[:, kvec[x] - 1] = FlipedGOmega[x]
        if lvec[x] > 0:
            mainmatrix[:, lvec[x] - 1] = FlipedGOmega[x]

    mainmatrix[rowi_sigma_i - 1, colj_sigma_i - 1] = 1.0
    return mainmatrix


def FlipedGOmegaFn_APseq_CL(M_org, N_org):
    num_omega = max(M_org, N_org) + 1

    sigma = 0.25
    difference = 4.0 / (num_omega - 1)

    GaussianSigma = np.zeros((1, num_omega - 1), dtype=float)
    for n in range(1, num_omega):
        GaussianSigma[0, n - 1] = 4.0 - n * difference

    GaussianSigma = GaussianSigma[:, :-1]

    GaussianOmega = sigma * GaussianSigma
    FlipedGOmega = np.sort(GaussianOmega, axis=1)[:, ::-1]

    sum_Fomega = np.sum(FlipedGOmega)
    FlipedGOmega = FlipedGOmega / sum_Fomega

    return FlipedGOmega


def build_node_list(main_nodes, neighbor_nodes):
    main_nodes = np.asarray(main_nodes).reshape(-1, order="F")
    neighbor_nodes = np.asarray(neighbor_nodes)

    num_main = main_nodes.size
    num_neighbors = neighbor_nodes.shape[1]

    node_list = np.zeros((num_main * num_neighbors, 2), dtype=int)
    q = 0

    for p in range(0, num_main * num_neighbors, num_neighbors):
        node_list[p:p + num_neighbors, 0] = main_nodes[q]
        node_list[p:p + num_neighbors, 1] = np.asarray(neighbor_nodes[q, :]).reshape(-1, order="F")
        q += 1

    return node_list


def nearest_matlab_like(edge_array, num_nodes, source, k):
    adj = [[] for _ in range(num_nodes + 1)]

    for a, b in edge_array:
        a = int(a)
        b = int(b)
        adj[a].append(b)
        adj[b].append(a)

    for node in range(1, num_nodes + 1):
        if adj[node]:
            adj[node] = sorted(set(adj[node]))

    dist = np.full(num_nodes + 1, np.inf)
    dist[source] = 0.0

    q = deque([source])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if np.isinf(dist[v]):
                dist[v] = dist[u] + 1.0
                q.append(v)

    candidates = []
    for node in range(1, num_nodes + 1):
        if node != source and np.isfinite(dist[node]):
            candidates.append((dist[node], node))

    candidates.sort(key=lambda t: (t[0], t[1]))
    return np.array([node for _, node in candidates[:k]], dtype=int)

# =========================================================
# EXACT CACHE BUILDER
# SAME MATH, JUST PRECOMPUTED ONCE
# =========================================================
def build_free_energy_cache(design_variables):
    n = int(round(np.sqrt(design_variables)))
    if n * n != design_variables:
        raise ValueError("design_variables must form a square lattice")

    M_org = n
    N_org = n
    Numel_m_size = M_org * N_org
    m = np.reshape(np.arange(1, Numel_m_size + 1), (M_org, N_org), order="F")

    num_center = (M_org - 2) * (N_org - 2)
    mle_node = np.zeros((num_center, 8), dtype=int)
    k1 = 0
    for i in range(1, M_org - 1):
        for j in range(1, N_org - 1):
            k1 += 1
            e1 = m[i, j + 1]
            e2 = m[i + 1, j]
            e3 = m[i - 1, j]
            e4 = m[i - 1, j - 1]
            e5 = m[i, j - 1]
            e6 = m[i + 1, j + 1]
            e7 = m[i - 1, j + 1]
            e8 = m[i + 1, j - 1]
            mle_node[k1 - 1, :] = [e1, e2, e3, e4, e5, e6, e7, e8]

    lc_node = np.zeros((M_org - 2, 5), dtype=int)
    k2 = 0
    for i in range(1, M_org - 1):
        k2 += 1
        le1 = m[i, 1]
        le2 = m[i + 1, 0]
        le3 = m[i + 1, 1]
        le4 = m[i - 1, 0]
        le5 = m[i - 1, 1]
        lc_node[k2 - 1, :] = [le1, le2, le3, le4, le5]

    rc_node = np.zeros((M_org - 2, 5), dtype=int)
    k3 = 0
    for i in range(1, M_org - 1):
        k3 += 1
        re1 = m[i, N_org - 2]
        re2 = m[i + 1, N_org - 1]
        re3 = m[i - 1, N_org - 1]
        re4 = m[i - 1, N_org - 2]
        re5 = m[i + 1, N_org - 2]
        rc_node[k3 - 1, :] = [re1, re2, re3, re4, re5]

    br_node = np.zeros((N_org - 2, 5), dtype=int)
    k4 = 0
    for j in range(1, N_org - 1):
        k4 += 1
        be1 = m[M_org - 1, j + 1]
        be2 = m[M_org - 1, j - 1]
        be3 = m[M_org - 2, j]
        be4 = m[M_org - 2, j - 1]
        be5 = m[M_org - 2, j + 1]
        br_node[k4 - 1, :] = [be1, be2, be3, be4, be5]

    tr_node = np.zeros((N_org - 2, 5), dtype=int)
    k5 = 0
    for j in range(1, N_org - 1):
        k5 += 1
        te1 = m[0, j - 1]
        te2 = m[1, j]
        te3 = m[0, j + 1]
        te4 = m[1, j + 1]
        te5 = m[1, j - 1]
        tr_node[k5 - 1, :] = [te1, te2, te3, te4, te5]

    cleftop = np.array([m[0, 1], m[1, 0], m[1, 1]], dtype=int)
    cleftbottom = np.array([m[M_org - 2, 0], m[M_org - 1, 1], m[M_org - 2, 1]], dtype=int)
    crhttop = np.array([m[0, N_org - 2], m[1, N_org - 1], m[1, N_org - 2]], dtype=int)
    crhtbottom = np.array([m[M_org - 2, N_org - 1], m[M_org - 1, N_org - 2], m[M_org - 2, N_org - 2]], dtype=int)

    row1 = np.array([
        1, 1, 1,
        m[M_org - 1, 0], m[M_org - 1, 0], m[M_org - 1, 0],
        m[0, N_org - 1], m[0, N_org - 1], m[0, N_org - 1],
        m[M_org - 1, N_org - 1], m[M_org - 1, N_org - 1], m[M_org - 1, N_org - 1]
    ], dtype=int)

    row2 = np.concatenate([cleftop, cleftbottom, crhttop, crhtbottom]).astype(int)
    node_corner = np.vstack([row1, row2]).T

    mcenter = m[1:M_org - 1, 1:N_org - 1].T
    mcenter = np.reshape(mcenter, (1, mcenter.size), order="F")
    node_center = build_node_list(mcenter, mle_node)

    m_lc = m[1:M_org - 1, 0]
    node_lc = build_node_list(m_lc.reshape(-1, order="F"), lc_node)

    m_rc = m[1:M_org - 1, N_org - 1]
    node_rc = build_node_list(m_rc.reshape(-1, order="F"), rc_node)

    m_br = m[M_org - 1, 1:N_org - 1]
    node_br = build_node_list(m_br.reshape(-1, order="F"), br_node)

    m_tr = m[0, 1:N_org - 1]
    node_tr = build_node_list(m_tr.reshape(-1, order="F"), tr_node)

    node_f = np.vstack([node_corner, node_tr, node_br, node_rc, node_lc, node_center]).astype(int)

    M_nf, N_nf = node_f.shape
    for l in range(M_nf):
        r = node_f[l, 0]
        c = node_f[l, 1]
        dx = np.where(node_f[:, 0] == c)[0]

        for ll in range(dx.size):
            linear_idx_1based = dx[ll] + 1
            row = ((linear_idx_1based - 1) % M_nf) + 1
            column = ((linear_idx_1based - 1) // M_nf) + 1

            if node_f[row - 1, column - 1] == c:
                if column < N_nf and node_f[row - 1, column] == r:
                    node_f[row - 1, column - 1] = 0
                    node_f[row - 1, column] = 0

    node_f = node_f[np.any(node_f != 0, axis=1), :]
    node_f = node_f[:, np.any(node_f != 0, axis=0)]

    FlipedGOmega = FlipedGOmegaFn_APseq_CL(M_org, N_org).reshape(-1)

    wsnum = Numel_m_size - 1
    nodeIDs = []
    for nID in range(1, Numel_m_size + 1):
        nodeIDs.append(nearest_matlab_like(node_f, Numel_m_size, nID, wsnum))

    r_of = np.zeros(Numel_m_size, dtype=int)
    c_of = np.zeros(Numel_m_size, dtype=int)
    for idx in range(1, Numel_m_size + 1):
        r_of[idx - 1] = ((idx - 1) % M_org) + 1
        c_of[idx - 1] = ((idx - 1) // M_org) + 1

    neighbor_idx0 = []
    neighbor_weights = []

    for nID in range(1, Numel_m_size + 1):
        ri = r_of[nID - 1]
        cj = c_of[nID - 1]

        neigh = nodeIDs[nID - 1]
        idx0 = neigh - 1

        rr = r_of[idx0]
        cc = c_of[idx0]

        cheb_dist = np.maximum(np.abs(rr - ri), np.abs(cc - cj))
        weights = FlipedGOmega[cheb_dist - 1]

        neighbor_idx0.append(idx0)
        neighbor_weights.append(weights.astype(float))

    return {
        "n": n,
        "M_org": M_org,
        "N_org": N_org,
        "Numel_m_size": Numel_m_size,
        "neighbor_idx0": neighbor_idx0,
        "neighbor_weights": neighbor_weights,
    }


# =========================================================
# MAIN VERSION
# =========================================================
def FreeEnergy_stochastic_same_exact_CL_fast(x_spin, x, y, cache):
    x = np.asarray(x).reshape(-1, order="F")
    y = np.asarray(y).reshape(-1, order="F")

    x_spin = np.asarray(x_spin).reshape(-1, order="F")
    x_spin = x_spin > 0

    n = np.sqrt(x_spin.size)
    if abs(n - round(n)) > np.finfo(float).eps:
        raise ValueError("x_spin must form a square lattice")
    n = int(round(n))

    if n != cache["n"]:
        raise ValueError("cache size does not match x_spin lattice size")

    mainmatrix = x_spin.astype(int) + 1
    mainmatrix[mainmatrix > 1] = -1
    mainmatrix = np.reshape(mainmatrix, (n, n), order="F")

    sigVec = mainmatrix.reshape(-1, order="F").astype(float)

    Nsites = cache["Numel_m_size"]
    E0 = np.zeros(Nsites, dtype=float)

    for nID in range(Nsites):
        idx0 = cache["neighbor_idx0"][nID]
        w = cache["neighbor_weights"][nID]
        si = sigVec[nID]
        E0[nID] = np.sum(-(si * sigVec[idx0] * w))

    beta = 1.0 / y
    Henergy = E0[:, None] - sigVec[:, None] * x[None, :]
    RHS = np.exp(-Henergy * beta[None, :])
    Free_energy = (-(np.log(np.sum(RHS, axis=0))) / beta) / Nsites

    return float(np.std(Free_energy, ddof=1))


# =========================================================
# WRAPPER
# =========================================================
def free_energy_stochastic_same_exact_wrapper_cl_fast(opt_space, cache, x, y):
    opt_space = np.asarray(opt_space)
    if opt_space.ndim == 1:
        opt_space = opt_space.reshape(1, -1)

    pop = opt_space.shape[0]
    std = np.zeros(pop, dtype=float)

    for i in range(pop):
        std[i] = FreeEnergy_stochastic_same_exact_CL_fast(opt_space[i, :], x, y, cache)

    return std


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    data_path = "data.mat"
    data = loadmat(data_path)

    x = np.asarray(data["x"]).reshape(-1, order="F")
    y = np.asarray(data["y"]).reshape(-1, order="F")

    ok, cache = validate_fast_vs_reference(
        x=x,
        y=y,
        n_vars=100,
        n_tests=100,
        seed=0,
        atol=1e-12
    )

    if not ok:
        raise RuntimeError("Fast function failed validation. Aborting.")

    config = {
        "numPopulation": 15,
        "maxGeneration": 100,
        "designVariables": 100,
        "typeOfOptimisation": "BINARY",
        "penaltyFactor": 25,
        "initialPenalty": 20,
        "deltaTheta": 0.45
    }

    print("Running BQPhy test optimization...")

    optimizer = qea.BQPhy_OPTIMISER()
    optimizer.initialize(config)

    optimizer.model(
        lambda opt_space: free_energy_stochastic_same_exact_wrapper_cl_fast(
            opt_space, cache, x, y
        )
    )

    t_start = time.time()
    optimizer.runOptimization()
    t_end = time.time()

    best_solution, fitness_history = optimizer.getBestDesign()

    print(f"Best solution: {best_solution}")
    print(f"Best fitness: {fitness_history[-1]:.12f}")
    print(f"Total optimization time: {t_end - t_start:.2f} seconds")