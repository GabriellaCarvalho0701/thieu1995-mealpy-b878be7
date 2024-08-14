import numpy as np
from opfunu.cec_based.cec2017 import F292017
from mealpy import GA
from mealpy import FloatVar, Problem

ndim = 30
f18 = F292017(ndim, f_bias=0)

def objective_function(solution):
    return f18.evaluate(solution)

bounds = FloatVar(lb=f18.lb, ub=f18.ub)
P1 = Problem(obj_func=objective_function, bounds=bounds, minmax="min")

epoch = 100
pop_size = 50
model = GA.EliteSingleGA(epoch=epoch, pop_size=pop_size)

if __name__ == "__main__":
    model.solve(P1)

    print("Bounds:", model.problem.bounds)
    print("Histórico:", model.history)
    print("Melhor Solução:", model.g_best.solution)
    print("Melhor Fitness:", model.g_best.target)