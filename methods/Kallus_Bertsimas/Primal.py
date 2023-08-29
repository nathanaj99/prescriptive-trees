'''
This script is for the decision-tree formulation.
'''

from gurobipy import *
import numpy as np


class Primal:
    def __init__(self, data, features, treatment_col, true_outcome_cols, outcome, bertsimas, n_min, tree, branching_limit,
                 time_limit):
        self.data = data
        self.treatment = treatment_col
        self.true_outcome_cols = true_outcome_cols
        self.outcome = outcome
        self.bertsimas = bertsimas
        self.n_min = n_min

        self.treatments_set = data[self.treatment].unique()  # set of possible treatments

        self.features = features

        self.tree = tree
        self.branching_limit = branching_limit

        # Decision Variables
        self.gamma = 0
        self.lamb = 0
        self.w = 0
        self.mu = 0
        self.nu = 0
        self.chi = 0
        if self.bertsimas:
            self.f = 0
            self.beta = 0
            self.theta = 0.5

        self.m = set(range(len(self.true_outcome_cols)))

        # define other variables
        self.train_y = self.data[self.outcome]
        self.train_X = self.data[self.features]
        self.train_t = self.data[self.treatment]
        self.n = len(self.train_X)

        # constants
        self.C = []
        for i in range(len(self.train_X.columns)):
            self.C.append((i, 0))

        # ---- Big M constraints ----
        self.minimum = min(self.train_y)

        self.ybar = self.train_y - self.minimum

        self.ymax = max(self.ybar)
        self.unique, self.counts = np.unique(self.train_t, return_counts=True)
        self.M = np.array(self.counts)
        self.M -= len(self.tree.Terminals) * self.n_min
        self.M = max(self.M)

        # ANCESTORS and RIGHT_LEFT
        self.ancestors = self.tree.ancestors_dic()
        self.right_left = self.tree.get_right_left()

        # Gurobi model
        self.model = Model('Kallus') # can I fix this?
        # self.model.params.Threads = 1
        self.model.params.TimeLimit = time_limit

    ###########################################################
    # Create the master problem
    ###########################################################
    def create_primal_problem(self):
        ############################### define variables

        self.gamma = self.model.addVars(self.tree.Nodes, len(self.C), vtype=GRB.BINARY, name='gamma')
        self.lamb = self.model.addVars(self.tree.Terminals, self.m, vtype=GRB.BINARY, name='lamb')
        self.w = self.model.addVars(self.n, self.tree.Terminals, lb=0, ub=1, name='w')
        self.mu = self.model.addVars(self.tree.Terminals, lb=0, name='mu')
        self.nu = self.model.addVars(self.n, self.tree.Terminals, lb=0, name='nu')
        self.chi = self.model.addVars(self.tree.Nodes, self.n, vtype=GRB.BINARY, name='chi')
        if self.bertsimas:
            self.f = self.model.addVars(self.n, lb=0, name='f')
            self.beta = self.model.addVars(self.tree.Terminals, self.m, lb=0, name='beta')

        ############################### define constraints

        # z[i,n] = z[i,l(n)] + z[i,r(n)] + zeta[i,n]    forall i, n in Nodes

        for p in self.tree.Nodes:
            self.model.addConstr(quicksum(self.gamma[p, i] for i in range(len(self.C))) == 1)

        # add chi constraint
        for i in range(self.n):
            for p in self.tree.Nodes:
                self.model.addConstr(
                    self.chi[p, i] == quicksum(self.gamma[p, j] for j in range(len(self.C)) if self.C[j][1] >=
                                               self.train_X.iloc[i, self.C[j][0]]))

        # add membership restriction from its ancestors (4d&e)
        for p in self.tree.Terminals:
            A_p = self.ancestors[p]  # index ancestors of p
            for q in A_p:
                R_pq = self.right_left[(p, q)]
                for i in range(self.n):
                    self.model.addConstr(self.w[i, p] <= (1 + R_pq) / 2 - R_pq * self.chi[q, i])

        #4e
        for p in self.tree.Terminals:
            A_p = self.ancestors[p]  # index ancestors of p
            for i in range(self.n):
                self.model.addConstr(self.w[i, p] >= 1 + quicksum(-self.chi[q, i] for q in A_p if self.right_left[
                    (p, q)] == 1) + quicksum(-1 + self.chi[q, i] for q in A_p if self.right_left[(p, q)] == -1))

        #4f
        for t in self.m:
            for p in self.tree.Terminals:
                self.model.addConstr(quicksum(self.w[i, p] for i in range(self.n) if self.train_t[i] == t) >= self.n_min)

        # Constraints 4g&h (Linearization of nu)
        # CHECKED
        for p in self.tree.Terminals:
            for i in range(self.n):
                self.model.addConstr(self.nu[i, p] <= self.ymax * self.w[i, p])
                self.model.addConstr(self.nu[i, p] <= self.mu[p])
                self.model.addConstr(self.nu[i, p] >= self.mu[p] - self.ymax * (1 - self.w[i, p]))

        # Constraint 4i (Choice of treatment applied to p)
        # CHECKED
        for p in self.tree.Terminals:
            self.model.addConstr(quicksum(self.lamb[p, t] for t in self.m) == 1)

        # ADD CONSTRAINT NOT ALLOWING TREATMENT OF K=2
        # for p in self.tree.Terminals:
        #     self.lamb[p, 2].lb = 0
        #     self.lamb[p, 2].ub = 0

        # Constraint 4j&k (Consistency between lambda and mu)
        # CHECKED. There are some inconsistencies where some w don't appear, but this is because ybar is 0 (i.e. the minimum)
        for p in self.tree.Terminals:
            for t in self.m:
                self.model.addConstr(
                    quicksum(self.nu[i, p] - self.w[i, p] * self.ybar[i] for i in range(self.n) if self.train_t[i] == t) <= self.M * (
                                1 - self.lamb[p, t]))
                self.model.addConstr(
                    quicksum(self.nu[i, p] - self.w[i, p] * self.ybar[i] for i in range(self.n) if self.train_t[i] == t) >= self.M * (
                                self.lamb[p, t] - 1))

        if self.bertsimas:
            for i in range(self.n):
                for p in self.tree.Terminals:
                    for t in self.m:
                        if self.train_t[i] == t:
                            self.model.addConstr(self.f[i] - self.beta[p, t] <= self.M * (1 - self.w[i, p]))
                            self.model.addConstr(self.f[i] - self.beta[p, t] >= self.M * (self.w[i, p] - 1))

        # define objective function
        if self.bertsimas:
            self.model.setObjective(
                self.theta * quicksum(self.nu[i, p] for i in range(self.n) for p in self.tree.Terminals) - (1 - self.theta) *
                quicksum((self.train_y[i] - self.f[i]) * (self.train_y[i] - self.f[i]) for i in range(self.n)),
                GRB.MAXIMIZE)
        else:
            self.model.setObjective(quicksum(self.nu[i, p] for i in range(self.n) for p in self.tree.Terminals),
                                    GRB.MAXIMIZE)