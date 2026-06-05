import numpy as np
import cvxpy as cp

class MPCController:
    def __init__(self, model, Np=10, Nc=5, Q_diag=None, R_diag=None):
        self.model = model
        self.A = model.A
        self.B = model.B
        self.Np = Np
        self.Nc = Nc
        self.nx = model.nx
        self.nu = model.nu
        if Q_diag is None:
            self.Q = np.diag([100.0,50.0,200.0,20.0])
        else:
            q = np.array(Q_diag, dtype=float)
            if q.size == self.nx:
                self.Q = np.diag(q)
            else:
                self.Q = np.diag(np.resize(q, self.nx))
        self.R = np.diag(R_diag if R_diag is not None else [0.1]*self.nu)
        self.u_min = np.zeros(self.nu)
        self.u_max = np.ones(self.nu)

    def solve(self, x0, x_ref_traj=None):
        Np = self.Np; nx = self.nx; nu = self.nu; A = self.A; B = self.B
        if x_ref_traj is None:
            x_ref_traj = [self.model.x_base.copy() for _ in range(Np)]
        U = cp.Variable((nu, Np))
        X = cp.Variable((nx, Np+1))
        cost = 0.0; constraints = [X[:,0] == x0]
        for k in range(Np):
            xref = x_ref_traj[k]
            dx = X[:,k] - xref
            cost += cp.quad_form(dx, self.Q)
            uk = U[:,k]; cost += cp.quad_form(uk, self.R)
            constraints += [X[:,k+1] == A @ X[:,k] + B @ U[:,k]]
            constraints += [U[:,k] >= self.u_min, U[:,k] <= self.u_max]
        dxN = X[:,Np] - x_ref_traj[-1]; cost += cp.quad_form(dxN, self.Q * 0.1)
        prob = cp.Problem(cp.Minimize(cost), constraints)
        try:
            prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)
        except Exception:
            prob.solve(solver=cp.SCS, verbose=False)
        if U.value is None:
            return np.ones(nu)*0.3
        u0 = np.array(U.value[:,0]).flatten()
        return np.clip(u0,0.0,1.0)
