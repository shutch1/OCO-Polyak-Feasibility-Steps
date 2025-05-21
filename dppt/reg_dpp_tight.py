#%% imports and functions
import numpy as np
import math
import cvxpy as cp
import time

## helper functions
def find_subg(A, x):
    max_ind = np.argmax(A @ x)
    return A[[np.max(max_ind)],:].T

## algorithm functions
def dpp(grad, g, A, x_past, Q_past, V, alpha, rho):
    # runs update for drift-plus-penalty
    g_val = np.max(g)
    s = find_subg(A, x_past)
    d = V*grad + s*Q_past
    proj_point = x_past - d/(2*alpha)
    proj_point_norm = np.linalg.norm(proj_point)
    if proj_point_norm == 0:
        x_new = proj_point
    else:
        x_new = min(xlim/proj_point_norm,1)*proj_point
    Q_new = max(Q_past + rho + g_val + s.T @ (x_new - x_past),0)

    return (x_new, Q_new)



#%% main code

prob_rng = np.random.default_rng(seed=0)
alg_rng = np.random.default_rng(seed=0)

# debug parameters
print_interval = 2000
num_trials = 30
save = True
losses = num_trials*[None]

# duration setup
T_max = int(2e4)
spacing = int(2e3)
N = math.floor(T_max/spacing)

# result arrays
reg_dpp = np.zeros((num_trials,N))
viol_dpp = np.zeros((num_trials,N))
true_viols = np.zeros((num_trials,N,4))

# run trials
for trial in range(num_trials):
    print('Starting Trial', trial)

    # problem parameters
    d = 2
    n = 4
    b = 0.5*np.ones((n,1))
    A = np.concatenate((np.eye(d),-np.eye(d)),axis=0)#np.array([[1, 1],[1, -1],[-1, 1],[-1, -1]]) #-1*np.ones((1,d)) #
    G_scal = 1
    sigma = 0
    epsilon = 0.45
    thetas = G_scal*np.concatenate((prob_rng.uniform(-1, 0,(T_max,1)), prob_rng.uniform(-1, 0,(T_max,1))),axis=1)
    eps = sigma*prob_rng.normal(0,1,(T_max,n))
    D = 2
    xlim = D/2
    cost_scal = 3
    G = cost_scal*(2*G_scal*math.sqrt(d)+2*xlim)
    S = math.sqrt(d)

    cost_func = lambda th, x : cost_scal*(x - th).T @ (x - th)
    cost_grad = lambda th, x : cost_scal*2*(x - th)

    start_time = time.time()
    for i in range(N):
        round_start_time = time.time()
        T = (i+1)*spacing

        # algorithm data for dpp
        x_dpp = np.zeros((d,1))
        alpha_scal = 1
        alpha_dpp = alpha_scal*T # these are the ones that are updated only once
        V_scal = 1
        V_dpp = V_scal*np.sqrt(T)
        Q_dpp = 0
        rho_scal = 20 #4.5
        rho_dpp = min(1,rho_scal/math.sqrt(T))*epsilon

        # init record variables
        theta_rec = np.zeros((T,d))
        cost_rec_dpp = np.zeros(T)
        viol_rec_dpp = np.zeros((T))
        sviol_rec_dpp = np.zeros((T))
        true_viol_rec = np.zeros((T,n))
        x_rec_dpp = np.zeros((T,d))
        expert_rec = np.zeros((T,d,2*d))
        weight_rec = np.zeros((T,2*d))
        Q_rec = np.zeros(T)

        # begin play
        for t in range(T):
            # choose cost function
            theta = thetas[[t],:].T

            # calculate gradient of cost
            grad_dpp = cost_grad(theta,x_dpp)

            # observe noisy constraint
            g_dpp = A @ x_dpp + eps[[t],:].T - b

            # record cost and violation
            theta_rec[t,:] = theta.T
            cost_rec_dpp[t] = cost_func(theta, x_dpp)
            viol_rec_dpp[t] = np.max(g_dpp)
            true_viol_rec[t,:] = g_dpp.T
            sviol_rec_dpp[t] = max(np.max(A @ x_dpp - b),0)
            x_rec_dpp[t,:] = x_dpp.T
            theta_rec[t,:] = theta.T
            # Q_rec[t] = Q_dpp

            (x_dpp, Q_dpp) = dpp(grad_dpp, g_dpp, A, x_dpp, Q_dpp, V_dpp, alpha_dpp, rho_dpp)

        # calculate regret for this round
        true_theta = np.sum(theta_rec,axis=0)/T
        x = cp.Variable((d,1))
        obj = cp.Minimize(cp.norm(x)**2 - 2*true_theta.T @ x)
        const = [A @ x <= b, cp.norm(x)<= xlim]
        prob = cp.Problem(obj, const)
        true_optim = cost_scal*(T*prob.solve() + np.sum(theta_rec**2))
        optim_point = x.value

        reg_dpp[trial,i] = np.sum(cost_rec_dpp) - true_optim
        viol_dpp[trial,i] = np.sum(viol_rec_dpp,axis=0)
        true_viols[trial,i,:] = np.sum(true_viol_rec,axis=0)

        print('Horizon',T,'DPP_Regret',np.round(reg_dpp[trial,i],3),'DPP_Viol',np.sum(viol_rec_dpp),'Time',np.round(time.time() - round_start_time,3), 'Time_per_round',(time.time() - round_start_time)/T)

    print('Total time',time.time() - start_time)

# save results
if save:
    tsteps = spacing*(np.arange(N)+1)
    np.save('tsteps_reg',tsteps)
    np.save('reg_dppt',reg_dpp)
    np.save('viol_dppt',viol_dpp)
    
# %%
