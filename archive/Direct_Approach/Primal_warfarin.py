'''
This script is for the decision-tree formulation.
'''

from gurobipy import *


class Primal:
    def __init__(self, data, features, protected_col, treatment_col, true_outcome_cols, outcome, regression, prob_t, robust, tree, branching_limit, fairness_bound,
                 time_limit):
        self.data = data
        self.datapoints = data.index
        self.treatment = treatment_col
        self.regression = regression
        self.protected_feature = protected_col
        self.true_outcome_cols = true_outcome_cols
        self.fairness_bound = fairness_bound
        self.outcome = outcome
        self.prob_t = prob_t
        self.robust = robust

        self.treatments_set = data[self.treatment].unique()  # set of possible treatments

        self.features = features

        self.tree = tree
        self.branching_limit = branching_limit

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

        # DON'T ALLOW ASSIGNING T=2
        for n in self.tree.Nodes + self.tree.Terminals:
             self.w[n, 2].ub = 0
             for i in self.datapoints:
                self.zeta[i, n, 2].ub = 0


        # ADD FAIRNESS CONSTRAINT
        # split by group membership

        protected_df = self.data[self.data[self.protected_feature] == 0]
        protected_prime_df = self.data[self.data[self.protected_feature] == 1]

        if self.fairness_bound <= 0.6:
            if self.robust:
                self.model.addConstr((1/len(protected_df) * (quicksum(self.zeta[i, n, k]*(protected_df.at[i, self.outcome] - protected_df.at[i, self.regression[int(k)]])/protected_df.at[i, self.prob_t] for i in protected_df.index for n in self.tree.Nodes + self.tree.Terminals for k in self.treatments_set if protected_df.at[i, self.treatment] == int(k)) + \
                    quicksum(self.zeta[i, n, k]*(protected_df.at[i, self.regression[int(k)]]) for i in protected_df.index for n in self.tree.Nodes + self.tree.Terminals for k in self.treatments_set))) - 
                        (1/len(protected_prime_df) * (quicksum(self.zeta[i, n, k]*(protected_prime_df.at[i, self.outcome] - protected_prime_df.at[i, self.regression[int(k)]])/protected_prime_df.at[i, self.prob_t] for i in protected_prime_df.index for n in self.tree.Nodes + self.tree.Terminals for k in self.treatments_set if protected_prime_df.at[i, self.treatment] == int(k)) + \
                    quicksum(self.zeta[i, n, k]*(protected_prime_df.at[i, self.regression[int(k)]]) for i in protected_prime_df.index for n in self.tree.Nodes + self.tree.Terminals for k in self.treatments_set))) <= self.fairness_bound
                        )

                self.model.addConstr((1/len(protected_prime_df) * (quicksum(self.zeta[i, n, k]*(protected_prime_df.at[i, self.outcome] - protected_prime_df.at[i, self.regression[int(k)]])/protected_prime_df.at[i, self.prob_t] for i in protected_prime_df.index for n in self.tree.Nodes + self.tree.Terminals for k in self.treatments_set if protected_prime_df.at[i, self.treatment] == int(k)) + \
                    quicksum(self.zeta[i, n, k]*(protected_prime_df.at[i, self.regression[int(k)]]) for i in protected_prime_df.index for n in self.tree.Nodes + self.tree.Terminals for k in self.treatments_set))) -
                        (1/len(protected_df) * (quicksum(self.zeta[i, n, k]*(protected_df.at[i, self.outcome] - protected_df.at[i, self.regression[int(k)]])/protected_df.at[i, self.prob_t] for i in protected_df.index for n in self.tree.Nodes + self.tree.Terminals for k in self.treatments_set if protected_df.at[i, self.treatment] == int(k)) + \
                    quicksum(self.zeta[i, n, k]*(protected_df.at[i, self.regression[int(k)]]) for i in protected_df.index for n in self.tree.Nodes + self.tree.Terminals for k in self.treatments_set))) <= self.fairness_bound
                        )
            else:
                self.model.addConstr(
                    (1/len(protected_df) * (quicksum(self.zeta[i, n, k]*(protected_df.at[i, self.regression[int(k)]]) for i in protected_df.index for n in self.tree.Nodes + self.tree.Terminals for k in self.treatments_set))) - 
                    (1/len(protected_prime_df) * (quicksum(self.zeta[i, n, k]*(protected_prime_df.at[i, self.regression[int(k)]]) for i in protected_prime_df.index for n in self.tree.Nodes + self.tree.Terminals for k in self.treatments_set))) <= self.fairness_bound
                        )

                self.model.addConstr(
                    (1/len(protected_prime_df) * (quicksum(self.zeta[i, n, k]*(protected_prime_df.at[i, self.regression[int(k)]]) for i in protected_prime_df.index for n in self.tree.Nodes + self.tree.Terminals for k in self.treatments_set))) -
                    (1/len(protected_df) * (quicksum(self.zeta[i, n, k]*(protected_df.at[i, self.regression[int(k)]]) for i in protected_df.index for n in self.tree.Nodes + self.tree.Terminals for k in self.treatments_set))) <= self.fairness_bound
                        )


        # PRESPECIFYING THE INNER NODES' VALUES
        # for n in self.tree.Nodes:
        #     # inner nodes cannot be treatment nodes
        #     self.p[n].ub = 0
        #     self.p[n].lb = 0

        #     # inner nodes cannot be treatment nodes
        #     for k in self.treatments_set:
        #         self.w[n, k].ub = 0

        #         for i in self.datapoints:
        #             self.zeta[i, n, k].ub = 0

        # define objective function
        obj = LinExpr(0)
        for i in self.datapoints:
            for n in self.tree.Nodes + self.tree.Terminals:
                for k in self.treatments_set:
                    obj.add(self.zeta[i, n, k]*(self.data.at[i, self.regression[int(k)]]))
                    treat = self.data.at[i, self.treatment]
                    if self.robust:
                        if int(treat) == int(k):
                            obj.add(self.zeta[i, n, k]*(self.data.at[i, self.outcome] - self.data.at[i, self.regression[int(k)]])/self.data.at[i, self.prob_t])

        self.model.setObjective(obj, GRB.MAXIMIZE)