'''
This script is for the decision-tree formulation.
'''

from gurobipy import *


class Primal:
    def __init__(self, data, features, dr_col, regression, robust, tree, branching_limit, time_limit, 
        fairness, budget_constraint):
        self.data = data
        self.datapoints = data.index
        self.dr_outcome = dr_col
        self.regression = regression
        self.robust = robust
        if fairness is not None:
            self.fairness_bound = fairness[0]
            self.protected_feature = fairness[1]
        else:
            self.fairness_bound = None
            self.protected_feature = None

        self.budget_constraint = budget_constraint

        self.treatments_set = range(len(self.regression))  # set of possible treatments

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

        # DON'T ALLOW ASSIGNING T=2 -- only for Warfarin
        if len(self.regression) > 2:
            for n in self.tree.Nodes + self.tree.Terminals:
                 self.w[n, 2].ub = 0
                 for i in self.datapoints:
                    self.zeta[i, n, 2].ub = 0


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

        if self.fairness_bound is not None:
            protected_df = self.data[self.data[self.protected_feature] == 0]
            protected_prime_df = self.data[self.data[self.protected_feature] == 1]
            if self.robust == 'direct':
                self.model.addConstr(
                    (1/(protected_df['count'].sum()) * (quicksum(self.zeta[i, n, k]*(protected_df.at[i, self.regression[int(k)]]) for i in protected_df.index for n in self.tree.Nodes + self.tree.Terminals for k in self.treatments_set))) - 
                    (1/(protected_prime_df['count'].sum()) * (quicksum(self.zeta[i, n, k]*(protected_prime_df.at[i, self.regression[int(k)]]) for i in protected_prime_df.index for n in self.tree.Nodes + self.tree.Terminals for k in self.treatments_set))) <= self.fairness_bound
                        )

                self.model.addConstr(
                    (1/(protected_prime_df['count'].sum()) * (quicksum(self.zeta[i, n, k]*(protected_prime_df.at[i, self.regression[int(k)]]) for i in protected_prime_df.index for n in self.tree.Nodes + self.tree.Terminals for k in self.treatments_set))) -
                    (1/(protected_df['count'].sum()) * (quicksum(self.zeta[i, n, k]*(protected_df.at[i, self.regression[int(k)]]) for i in protected_df.index for n in self.tree.Nodes + self.tree.Terminals for k in self.treatments_set))) <= self.fairness_bound
                        )

            elif self.robust == 'robust':
                self.model.addConstr(
                    (1/(protected_df['count'].sum()) * (quicksum(self.zeta[i, n, k]*((protected_df.at[i, self.regression[int(k)]]) + (protected_df.at[i, self.dr_outcome[int(k)]])) for i in protected_df.index for n in self.tree.Nodes + self.tree.Terminals for k in self.treatments_set))) - 
                    (1/(protected_prime_df['count'].sum()) * (quicksum(self.zeta[i, n, k]*((protected_prime_df.at[i, self.regression[int(k)]]) + (protected_prime_df.at[i, self.dr_outcome[int(k)]])) for i in protected_prime_df.index for n in self.tree.Nodes + self.tree.Terminals for k in self.treatments_set))) <= self.fairness_bound
                        )

                self.model.addConstr( 
                    (1/(protected_prime_df['count'].sum()) * (quicksum(self.zeta[i, n, k]*((protected_prime_df.at[i, self.regression[int(k)]]) + (protected_prime_df.at[i, self.dr_outcome[int(k)]])) for i in protected_prime_df.index for n in self.tree.Nodes + self.tree.Terminals for k in self.treatments_set))) -
                    (1/(protected_df['count'].sum()) * (quicksum(self.zeta[i, n, k]*((protected_df.at[i, self.regression[int(k)]]) + (protected_df.at[i, self.dr_outcome[int(k)]])) for i in protected_df.index for n in self.tree.Nodes + self.tree.Terminals for k in self.treatments_set))) <= self.fairness_bound
                        )

            else: # NOT TESTED
                self.model.addConstr(
                    (1/(protected_df['count'].sum()) * (quicksum(self.zeta[i, n, k]*(protected_df.at[i, self.dr_outcome[int(k)]]) for i in protected_df.index for n in self.tree.Nodes + self.tree.Terminals for k in self.treatments_set))) - 
                    (1/(protected_prime_df['count'].sum()) * (quicksum(self.zeta[i, n, k]*(protected_prime_df.at[i, self.dr_outcome[int(k)]]) for i in protected_prime_df.index for n in self.tree.Nodes + self.tree.Terminals for k in self.treatments_set))) <= self.fairness_bound
                        )

                self.model.addConstr(
                    (1/(protected_prime_df['count'].sum()) * (quicksum(self.zeta[i, n, k]*(protected_prime_df.at[i, self.dr_outcome[int(k)]]) for i in protected_prime_df.index for n in self.tree.Nodes + self.tree.Terminals for k in self.treatments_set))) -
                    (1/(protected_df['count'].sum()) * (quicksum(self.zeta[i, n, k]*(protected_df.at[i, self.dr_outcome[int(k)]]) for i in protected_df.index for n in self.tree.Nodes + self.tree.Terminals for k in self.treatments_set))) <= self.fairness_bound
                        )



        if self.budget_constraint is not None:
            for treatment, max_prop in self.budget_constraint.items():
                max_treated = max_prop * self.data['count'].sum()
                self.model.addConstr(
                    quicksum(
                        self.zeta[i, n, treatment]*self.data.at[i, 'count'] for i in self.datapoints for n in self.tree.Nodes + self.tree.Terminals) <= max_treated)


        # define objective function
        obj = LinExpr(0)
        for i in self.datapoints:
            for n in self.tree.Nodes + self.tree.Terminals:
                for k in self.treatments_set:
                    if self.robust == 'direct':
                        obj.add(self.zeta[i, n, k]*(self.data.at[i, self.regression[int(k)]]))
                    elif self.robust == 'robust':
                        obj.add(self.zeta[i, n, k]*((self.data.at[i, self.regression[int(k)]]) + (self.data.at[i, self.dr_outcome[int(k)]])))
                    else:
                        obj.add(self.zeta[i, n, k]*(self.data.at[i, self.dr_outcome[int(k)]]))


        self.model.setObjective(obj, GRB.MAXIMIZE)