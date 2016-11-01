import numpy as np

from messages import SetStepSize
import utils


class AdamOptimizer:
    """An optimizer using the Adam algorithm for stochastic approximation. Given the parameters
    array x, it takes scaled gradient descent steps when supplied with x's gradient. The step_size
    value controls the maximum amount a parameter may change per step. b1 and b2 are Adam momentum
    parameters which should not need to be changed from the default."""
    def __init__(self, x, opfunc, step_size=SetStepSize.default_adam, b1=0.9, b2=0.999):
        self.x = x
        self.opfunc = opfunc
        self.step_size = step_size
        self.b1 = b1
        self.b2 = b2
        self.t = 0
        self.g1 = np.zeros_like(x)
        self.g2 = np.zeros_like(x)

    def step(self):
        """Takes a scaled gradient descent step. Updates x in place, and returns the new value."""
        self.t += 1
        loss, grad = self.opfunc(self.x)

        self.g1[:] = self.b1*self.g1 + (1-self.b1)*grad
        self.g2[:] = self.b2*self.g2 + (1-self.b2)*grad**2
        ss = self.step_size * np.sqrt(1-self.b2**self.t) / (1-self.b1**self.t)

        self.x -= ss * self.g1 / (np.sqrt(self.g2) + 1e-8)
        return self.x, loss

    def resample(self, size, new_x=None):
        """Resamples the optimizer's internal state on the last two axes to a new HxW size."""
        if new_x is not None:
            self.x = new_x
            size = self.x.shape[2:]
        else:
            self.x = utils.resize(self.x, size)
        self.g2 = np.maximum(utils.resize(self.g2, size, order=1), 0)
        self.objective_changed()
        return self.x

    def objective_changed(self):
        """Advises the optimizer that the objective function has changed and that it should discard
        internal state as appropriate."""
        self.g1 = np.zeros_like(self.x)


class LBFGSOptimizer:
    def __init__(self, x, opfunc, step_size=SetStepSize.default_lbfgs, n_corr=10):
        """L-BFGS for function minimization, with fixed size steps (no line search)."""
        self.x = x
        self.opfunc = opfunc
        self.step_size = step_size
        self.n_corr = n_corr
        self.loss = None
        self.grad = None
        self.sk = []
        self.yk = []
        self.syk = []

    def step(self):
        """Take an L-BFGS step. Returns the new parameters and the loss after the step."""
        if self.loss is None:
            self.loss, self.grad = self.opfunc(self.x)

        # Compute and take an L-BFGS step
        s = -self.step_size * self.inv_hv(self.grad)
        self.x += s

        # Compute a curvature pair and store parameters for the next step
        loss, grad = self.opfunc(self.x)
        y = grad - self.grad
        self.store_curvature_pair(s, y)
        self.loss, self.grad = loss, grad

        return self.x, loss

    def store_curvature_pair(self, s, y):
        """Updates the L-BFGS memory with a new curvature pair."""
        sy = utils.dot(s, y)
        if sy > 1e-10:
            self.sk.append(s)
            self.yk.append(y)
            self.syk.append(sy)
        if len(self.sk) > self.n_corr:
            self.sk, self.yk, self.syk = self.sk[1:], self.yk[1:], self.syk[1:]

    def inv_hv(self, p):
        """Computes the product of a vector with an approximation of the inverse Hessian."""
        p = p.copy()
        alphas = []
        for s, y, sy in zip(reversed(self.sk), reversed(self.yk), reversed(self.syk)):
            alphas.append(utils.dot(s, p) / sy)
            utils.axpy(-alphas[-1], y, p)

        if len(self.sk) > 0:
            sy, y = self.syk[-1], self.yk[-1]
            p *= sy / utils.dot(y, y)
        else:
            # With no curvature information, take a reasonably-scaled step
            p /= np.sqrt(utils.dot(p, p) / p.size)

        for s, y, sy, alpha in zip(self.sk, self.yk, self.syk, reversed(alphas)):
            beta = utils.dot(y, p) / sy
            utils.axpy(alpha - beta, s, p)

        return p

    def resample(self, size, new_x=None):
        """Resamples the optimizer's internal state to a new HxW size. The L-BFGS memory is cleared
        and the Adam moment accumulators are resized."""
        if new_x is not None:
            self.x = new_x
            size = new_x.shape[-2:]
        else:
            self.x = utils.resize(self.x, size)
        self.objective_changed()
        return self.x

    def objective_changed(self):
        """Advises the optimizer that the objective function has changed and that it should discard
        internal state as appropriate."""
        self.sk, self.yk, self.syk = [], [], []
        self.loss, self.grad = None, None
