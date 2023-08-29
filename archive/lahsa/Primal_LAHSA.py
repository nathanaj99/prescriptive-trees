'''
This script is for the decision-tree formulation.
'''

from gurobipy import *


class Primal:
    def __init__(self, data, features, treatment_col, disab_col, pshTr, outcome, prob_t, tree, time_limit, method, y_hat, b_warm, budget_constraint, z_greedy):
        self.data = data
        self.datapoints = data.index
        self.treatment = treatment_col
        self.outcome = outcome
        self.prob_t = prob_t
        self.method = method
        self.y_hat = y_hat
        self.b_warm = b_warm
        self.z_greedy = z_greedy
        self.budget_constraint = budget_constraint
        self.disab = disab_col
        self.pshT = pshTr

        self.treatments_set = data[self.treatment].unique()  # set of possible treatments

        self.features = features

        self.tree = tree

        # Decision Variables
        self.b = 0
        self.p = 0
        self.w = 0
        self.z = 0
        self.zeta = 0

        # Gurobi model
        self.model = Model('FlowOPT')
        # self.model.params.Threads = 1
        self.model.params.TimeLimit = time_limit

        # NEW ADDITIONS TO SPEED UP SOLVER
        # Root and Node Method - Barrier (2) vs DualSimplex (1) to solve LP Relaxation at each node
        self.model.setParam('Method',2)
        self.model.setParam('NodeMethod',2)
        self.model.setParam('Crossover',0) # Cross Over speeds up solving for Barrier

    ###########################################################
    # Create the master problem
    ###########################################################
    def create_primal_problem(self):

        ############################### define variables

        self.b = self.model.addVars(self.tree.Nodes, self.features, vtype=GRB.BINARY, name='b')
        self.p = self.model.addVars(self.tree.Nodes + self.tree.Terminals, vtype=GRB.BINARY, name='p')
        self.w = self.model.addVars(self.tree.Nodes + self.tree.Terminals, self.treatments_set, vtype=GRB.CONTINUOUS, lb=0,
                                    name='w')
        self.zeta = self.model.addVars(self.datapoints, self.tree.Nodes + self.tree.Terminals, self.treatments_set,
                                       vtype=GRB.CONTINUOUS, lb=0, name='zeta')
        self.z = self.model.addVars(self.datapoints, self.tree.Nodes + self.tree.Terminals, vtype=GRB.CONTINUOUS, lb=0,
                                    name='z')

        ############################### define constraints

        ## ADDING THE WARM START BRANCHING VARIABLES
        # if self.b_warm is not None:
        #     self.model.addConstrs(self.b[k] == self.b_warm[k] for k in self.b_warm.keys())

        # z[i,n] = z[i,l(n)] + z[i,r(n)] + zeta[i,n]    forall i, n in Nodes
        for n in self.tree.Nodes:
            n_left = int(self.tree.get_left_children(n))
            n_right = int(self.tree.get_right_children(n))
            self.model.addConstrs(
                (self.z[i, n] == self.z[i, n_left] + self.z[i, n_right] + quicksum(self.zeta[i, n, k] for k in self.treatments_set)) for i in self.datapoints)


        # z[i,l(n)] <= sum(b[n,f], f if x[i,f]<=0)    forall i, n in Nodes
        for i in self.datapoints:
            self.model.addConstrs((self.z[i, int(self.tree.get_left_children(n))] <= quicksum(
                self.b[n, f] for f in self.features if self.data.at[i, f] <= 0)) for n in self.tree.Nodes)

        # z[i,r(n)] <= sum(b[n,f], f if x[i,f]=1)    forall i, n in Nodes
        for i in self.datapoints:
            self.model.addConstrs((self.z[i, int(self.tree.get_right_children(n))] <= quicksum(
                self.b[n, f] for f in self.features if self.data.at[i, f] == 1)) for n in self.tree.Nodes)

        # sum(b[n,f], f) + p[n] + sum(p[m], m in A(n)) = 1   forall n in Nodes
        self.model.addConstrs(
            (quicksum(self.b[n, f] for f in self.features) + self.p[n] + quicksum(
                self.p[m] for m in self.tree.get_ancestors(n)) == 1) for n in
            self.tree.Nodes)

        # p[n] + sum(p[m], m in A(n)) = 1   forall n in Terminals
        self.model.addConstrs(
            (self.p[n] + quicksum(
                self.p[m] for m in self.tree.get_ancestors(n)) == 1) for n in
            self.tree.Terminals)

        #self.model.addConstr(self.p[1] == 0)

        # sum(sum(b[n,f], f), n) <= branching_limit
        # self.model.addConstr(
        #     (quicksum(
        #         quicksum(self.b[n, f] for f in self.features) for n in self.tree.Nodes)) <= self.branching_limit)

        # zeta[i,n] <= w[n,T[i]] for all n in N+L, i
        for n in self.tree.Nodes + self.tree.Terminals:
            for k in self.treatments_set:
                self.model.addConstrs(
                    self.zeta[i, n, k] <= self.w[n, k] for i in self.datapoints)

        # sum(w[n,k], k in treatments) = p[n]
        self.model.addConstrs(
            (quicksum(self.w[n, k] for k in self.treatments_set) == self.p[n]) for n in
            self.tree.Nodes + self.tree.Terminals)

        # 2k
        """for n in self.tree.Nodes + self.tree.Terminals:
            for i in self.datapoints:
                self.model.addConstr(quicksum(self.zeta[i, n, k] for k in self.treatments_set) == self.p[n])"""

        for n in self.tree.Terminals:
            self.model.addConstrs(quicksum(self.zeta[i, n, k] for k in self.treatments_set) == self.z[i, n] for i in self.datapoints)

        self.model.addConstrs(self.z[i, 1] ==1 for i in self.datapoints)

        #self.model.addConstr(self.b[1, 'V1'] == 1)
        #self.model.addConstr(self.b[2, 'V2'] == 1)

        # add budget constraints
        for k in self.treatments_set:
            if int(k) != 0:
                max_treated = self.budget_constraint[int(k)] * len(self.data)
                self.model.addConstr(
                    quicksum(
                        self.zeta[i, n, k] for i in self.datapoints for n in self.tree.Nodes + self.tree.Terminals) <= max_treated)

        ########################
        # Disability Constraint #
        # instead of doing mode.addConstr, it is better from memory perspective to just set
        # both upper (ob) and lower (lb) bounds to 0. Here I set all zetas for nondisabled individuals
        # to 0 for pshTr [2,3]
        for k in self.pshT:
            for i in self.datapoints:
                for n in self.tree.Nodes + self.tree.Terminals:
                    if self.data.at[i, self.disab] == 0:
                        self.zeta[i, n, k].ub = 0
        ########################


        # PRESPECIFYING THE INNER NODES' VALUES
        if self.b_warm is not None:
            for k, v in self.b_warm.items():
                if v == 0:
                    self.b[k].ub = 0
                    self.b[k].lb = 0
                else:
                    self.b[k].ub = 1
                    self.b[k].lb = 1

        if self.z_greedy is not None:
            for k, v in self.z_greedy.items():
                if int(k[0]) in self.datapoints:
                    self.z[k].ub = int(v)
                    self.z[k].lb = int(v)

        for n in self.tree.Nodes:
            # inner nodes cannot be treatment nodes
            self.p[n].ub = 0
            self.p[n].lb = 0

            # inner nodes cannot be treatment nodes
            for k in self.treatments_set:
                self.w[n, k].ub = 0

                for i in self.datapoints:
                    self.zeta[i, n, k].ub = 0


        # SOS Constraint - Specify at most one variable in a set can be a value other than 0
        for n in self.tree.Nodes:
            self.model.addSOS(GRB.SOS_TYPE1, [self.b[b] for b in self.b if b[0] == n]) # Only one of the features at each splitting node can be non-zero

        # define objective function
        obj = LinExpr(0)
        for i in self.datapoints:
            for n in self.tree.Nodes + self.tree.Terminals:
                for k in self.treatments_set:
                    if self.method == 'DM':
                        obj.add(self.zeta[i, n, k]*(self.data.at[i, self.y_hat[int(k)]]))
                    elif self.method == 'DR':
                        obj.add(self.zeta[i, n, k]*(self.data.at[i, self.y_hat[int(k)]]))
                        treat = self.data.at[i, self.treatment]
                        if int(treat) == int(k):
                            obj.add(self.zeta[i, n, k]*(self.data.at[i, self.outcome] - self.data.at[i, self.y_hat[int(k)]]) * self.data.at[i, self.prob_t])
                    elif self.method == 'IPW':
                        treat = self.data.at[i, self.treatment]
                        if int(treat) == int(k):
                            obj.add(self.zeta[i, n, k]*(self.data.at[i, self.outcome])*self.data.at[i, self.prob_t])

        # NEED TO CHANGE MULTIPLICATION TO DIVISION OF IPW

        self.model.setObjective(obj, GRB.MAXIMIZE)