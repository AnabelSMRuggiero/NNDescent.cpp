#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 10:49:27 2021

@author: anabel
"""
import math
import subprocess

import numpy as np

from pymoo.factory import get_algorithm, get_crossover, get_mutation, get_sampling, get_termination, get_selection
from pymoo.optimize import minimize
from pymoo.model.problem import Problem
from pymoo.visualization.scatter import Scatter


def RunNNDescent(blockNeighbors, nearestNodeNeighbors, queryDepth, targetSplitSize, searchNeighbors, searchDepth, searchQueue):
    comNeighbors = nearestNodeNeighbors + 5
    minSplitSize = math.floor(targetSplitSize * 0.6)
    maxSplitSize = math.ceil(targetSplitSize * 1.4)
    binArgs = "-blockGraphNeighbors=" + str(blockNeighbors)
    binArgs += " -COMNeighbors=" + str(comNeighbors)
    binArgs += " -nearestNodeNeighbors=" + str(nearestNodeNeighbors)
    binArgs += " -queryDepth=" + str(queryDepth)
    binArgs += " -targetSplitSize=" + str(targetSplitSize)
    binArgs += " -minSplitSize=" + str(minSplitSize)
    binArgs += " -maxSplitSize=" + str(maxSplitSize)
    binArgs += " -searchNeighbors=" + str(searchNeighbors)
    binArgs += " -searchDepth=" + str(searchDepth)
    binArgs += " -maxSearchesQueued=" + str(searchQueue)
    
    results = [0,0,0]
    #for i in range(5):    
    output = subprocess.run("./Bin/nndescent " + binArgs, shell=True, cwd="../", capture_output = True, text=True)
    print(output)
    splitOut =  output.stdout.split("\n")
    tmpResults = [float(x) for x in splitOut[0:-1]]
    results[0] += tmpResults[0]
    results[1] += tmpResults[1]
    results[2] += tmpResults[2]
        
    #results[0] /= 5
    #results[1] /= 5
    #results[2] /= 5
    return results

def PrintCommand(blockNeighbors, nearestNodeNeighbors, queryDepth, targetSplitSize, searchNeighbors, searchDepth, searchQueue):
    comNeighbors = nearestNodeNeighbors + 5
    minSplitSize = math.floor(targetSplitSize * 0.6)
    maxSplitSize = math.ceil(targetSplitSize * 1.4)
    binArgs = "-blockGraphNeighbors=" + str(blockNeighbors)
    binArgs += " -COMNeighbors=" + str(comNeighbors)
    binArgs += " -nearestNodeNeighbors=" + str(nearestNodeNeighbors)
    binArgs += " -queryDepth=" + str(queryDepth)
    binArgs += " -targetSplitSize=" + str(targetSplitSize)
    binArgs += " -minSplitSize=" + str(minSplitSize)
    binArgs += " -maxSplitSize=" + str(maxSplitSize)
    binArgs += " -searchNeighbors=" + str(searchNeighbors)
    binArgs += " -searchDepth=" + str(searchDepth)
    binArgs += " -maxSearchesQueued=" + str(searchQueue)
    print(binArgs)

class MyProblem(Problem):

    def __init__(self):
        super().__init__(n_var=6, n_obj=2, n_constr=0, xl=[5,3,1,60,1,1], xu=[25,40,10,480,10,20], type_var=int, elementwise_evaluation = True)

    def _evaluate(self, x, out, *args, **kwargs):
        print(x)
        runResult = RunNNDescent(x[0],x[1],x[2],x[3],max(10, x[5]),x[4], x[5])
        out["F"] = np.array([runResult[1], 100-runResult[2]])
        

problem = MyProblem()

method = get_algorithm("ga",
                       pop_size=20,
                       sampling=get_sampling("int_random"),
                       crossover=get_crossover("int_sbx", prob=1.0, eta=3.0),
                       mutation=get_mutation("int_pm", eta=3.0),
                       selection = get_selection("random"),
                       survival=None,
                       eliminate_duplicates=True)


termination = get_termination("time", "8:00:00")

results = minimize(problem, method, termination)

plot = Scatter(labels = ["Search Time (s)", "100 - Recall%"])
plot.add(results.F, color="red")
plot.show()
    
    
#test = RunNNDescent(10, 25, 4, 140)
    
    
    