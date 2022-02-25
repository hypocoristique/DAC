#python3

import numpy as np
import random
import math
from icecream import ic

def read_file(filename):
    with open(filename, 'r') as f:
        ctx = []
        rwd = []
        for l in f.readlines():
            id, c, r = l.split(":")
            ctx.append([float(x) for x in c.split(";")])
            rwd.append([float(x) for x in r.split(";")])
    return ctx, rwd

ctx, rwd = read_file('/Users/thomasfloquet/Desktop/Thomas/Etudes/Ecole/Sorbonne/RLD/TME/TME1/CTR.txt')

ctx = np.array(ctx)
rwd = np.array(rwd)
nb_articles, arms = rwd.shape
d = ctx.shape[1]

class RandomAgent:
    def __init__(self, nb_bras: int, dim:int):
        self.nb_bras = nb_bras
        self.dim = dim
        self.t = []
        self.r = []
        self.time_step = 0
    def act(self, ctx):
        arm = random.randint(0, self.nb_bras-1)
        self.t.append(arm)
        self.time_step += 1
        return arm
    def update(self, arm, rwd):
        reward = rwd[self.time_step-1,arm]
        self.r.append(reward)
        return reward

R = 0
agent = RandomAgent(10,5)
for c in range(nb_articles):
    idx = agent.act(ctx)
    R += agent.update(idx, rwd)
ic(R)


def greedy(init, rwd):
    R = np.zeros((init, arms))
    for arm in range(arms):
        for article in range(init):
            R[article, arm] = rwd[article, arm]
    R = np.sum(R, axis=1)
    arm = np.where(R==np.max(R))
    arm = arm[0][0]
    sum = np.sum(rwd[:,arm])
    arm += 1
    return arm, sum

arm, sum = greedy(10,rwd)
ic(sum)

def optimale(rwd):
    R = 0
    # history = []
    for i in range(nb_articles):
        R += np.max(rwd[i,:])
        # history.append(np.where(rwd[i,:]==max(rwd[i,:]))[0][0])
    return R

ic(optimale(rwd))

def UCB(init, rwd):
    R = 0
    T = [0]*arms
    g = [0]*arms
    policy = [0]*arms
    Init = np.zeros((init, arms))
    for article in range(init):
        for arm in range(arms):
            Init[article, arm] = rwd[article, arm]
        R += max(Init[article,:])
        arm = np.where(Init[article,:]==max(Init[article,:]))[0][0]
        T[arm] += 1
        g[arm] += R
    arm_notnull = [idx for idx, x in enumerate(T) if x != 0] #Je ne suis pas sûr d'avoir bien compris comment initialiser UCB. J'ai exploré chaque bras pour les init premiers articles puis ai conservé uniquement les bras qui avaient au moins une valeur max pour un des init premiers articles
    for t in range(init,nb_articles):
        for arm in arm_notnull:
            policy[arm] = (1/T[arm])*g[arm]+ math.sqrt((2*math.log(t))/T[arm])
        arm = policy.index(max(policy))
        T[arm] += 1
        g[arm] += rwd[t,arm]
        R += rwd[t,arm]
    return R

ic(UCB(10,rwd))

def linUCB(alpha, rwd):
    A = np.zeros((d, d*arms))
    b = np.zeros((d,arms))
    p = np.zeros((arms))
    first = []
    R = 0
    for article in range(nb_articles):
        for arm in range(arms):
            x = np.reshape(ctx[article,:], (-1,1))
            if arm not in first:
                A[:d,d*arm:d*arm+d] = np.eye(d)
                first.append(arm)
            invA = np.linalg.inv(A[:d,d*arm:d*arm+d])
            theta = np.dot(invA,b[:d,arm])
            p[arm] = np.dot(theta.T,x)[0] + alpha*math.sqrt(np.dot(np.dot(x.T,invA),x)[0][0])
        arm = np.where(p==max(p))[0][0]
        R += rwd[article,arm]
        A[:d,d*arm:d*arm+d] += np.dot(x,x.T)
        b[:d,arm] += np.reshape(np.dot(rwd[article,arm],x), b[:d,arm].shape)
    return R



ic(linUCB(0.15, rwd))


#Autre possible initialisation où on regarde les rewards des 10 bras pour le premier article (mais les résultats sont moins bons, ce qui est logique car on fait moins d'exploration) :

def UCB_1init(rwd):
    R = 0
    T = [0]*arms
    g = [0]*arms
    policy = [0]*arms
    for arm in range(arms):
        g[arm] = rwd[0, arm]
        T[arm] += 1
    R += max(g)
    arm = g.index(max(g))
    T[arm] += 1
    for t in range(1,nb_articles):
        for arm in range(arms):
            policy[arm] = (1/T[arm])*g[arm]+ math.sqrt((2*math.log(t))/T[arm])
        arm = policy.index(max(policy))
        T[arm] += 1
        g[arm] += rwd[t,arm]
        R += rwd[t,arm]
    return R

ic(UCB_1init(rwd))