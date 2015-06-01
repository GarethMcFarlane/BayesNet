__authors__ = 'Gareth McFarlane, Yoshi Hemzal'

import sys
import itertools
import copy
import random
import numpy as np

class Node:
    def __init__(self, name):
        self.name = name
        self.probabilities = None
        self.parent = []
        self.child = []

    def addchild(self, c):
        self.child.append(c)

    def addparnt(self, p):
        self.parent.append(p)

    def remchild(self, c):
        for c in self.child:
            if c.name == c:
                self.child.remove(c)
                return True
        return False

    def remparnt(self, p):
        for p in self.parent:
            if p.name == p:
                self.parent.remove(p)
                return True
        return False

    def getchildren(self):
        return self.child

    def getparents(self):
        return self.parent

    def initprobs(self):
        totalparents = len(self.parent)
        if totalparents is 0:
            self.probabilities = {'prob':None}
            return True

        totalcombinations = map(''.join, itertools.product("10", repeat=totalparents))
        self.probabilities = {}
        for t in totalcombinations:
            self.probabilities[t] = None
        return True

    def setcondprobs(self, probs):
        if not 'prob' in probs: return False
        if not self.parent:
            self.probabilities = {'prob':float(probs['prob'])}
            return True

        id = ""
        for p in self.parent:
            id = id + str(probs[p.name])
        self.probabilities[id] = float(probs['prob'])
        return True


class BayesNet:
    def __init__(self):
        self.nodes = []
        self.probs = []

    def clonenet(self):
        net = BayesNet()
        for i in self.nodes:
            net.addnode(i.name)
        for j in self.nodes:
            for c in j.children:
                net.connection(j.name, c.name)
        for k in self.nodes:
            net.getnode(k.name).probabilities = copy.deepcopy(k.probabilities)

        return net

    def topsort(self):
        sorted = []
        net = self.clonenet()

        rootnode = net.getrootnode()

        # should be only one here.
        while rootnode:
            scratchnode = rootnode.pop()
            sorted.append(scratchnode)
            for child in list(scratchnode.child):
                net.disconnection(scratchnode.name, child.name)
                if not child.parent:
                    rootnode.append(child)
        final = []
        for node in sorted:
            final.append(self.getnode(node.name))
        return final

    def addnode(self, nodename):
        self.nodes.append(Node(nodename))

    def getnode(self, nodename):
        for i in self.nodes:
            if i.name is nodename:
                return i
        return False

    def getrootnode(self):
        rootarr = []
        for n in self.nodes:
            if not n.parents:
                rootarr.append(n)
        return rootarr

    def connection(self, parentnode, childnode):
        parent = self.getnode(parentnode)
        child = self.getnode(childnode)

        if (parent, child) is not False:
            parent.addchild(child)
            child.addparnt(parent)
            return True
        return False

    def disconnection(self, parentnode, childnode):
        parent = self.getnode(parentnode)
        child = self.getnode(childnode)

        if (parent, child) is not False:
            parent.remparnt(parent)
            child.remchild(child)
            return True
        return False

    def initprobsnew(self):
        for node in self.nodes:
            node.initprobs()
        return True

    def addprobs(self, nodename, probs):
        return self.getnode(nodename).setcondprobs(probs)

    def getweight(self, nodes, evidence, outcome):
        resarray = {}
        initweight = 1.0
        good = True

        for n in nodes:
            if len(n.probabilities) is not 1:
                id = ""
                for p in n.parents:
                    id = id+str(resarray[p.name])
                if n.name in evidence:
                    if evidence[n.name] is 1:
                        resarray[n.name] = 1
                        initweight = initweight * n.probabilities[id]
                    else:
                        resarray[n.name] = 0
                        initweight = initweight * (1 - n.probabilities[id])
                else:
                    if (random.random() <= n.probabilities[id]):
                        resarray[n.name] = 1
                    else:
                        resarray[n.name] = 0
            else:
                if n.name in evidence:
                    if evidence[n.name] is 1:
                        resarray[n.name] = 1
                        initweight = initweight * n.probabilities['prob']
                    else:
                        resarray[n.name] = 0
                        initweight = initweight * (1.0 - n.probabilities['prob'])

                else:
                    if (random.random() <= n.probabilities['prob']):
                        resarray[n.name] = 1
                    else:
                        resarray[n.name] = 0

            if ((good is True) and (n.name in outcome)):
                if (outcome[n.name] != resarray[n.name]):
                    good = False

        return good, initweight

    def likelihoodweighting(self, evidence, outcome, samplenum):
        sorted = self.topsort()

        totalweight = 0.0
        condweight = 0.0001
        for x in xrange(0, samplenum):
            good, initweight = self.getweight(sorted, evidence, outcome)
            totalweight += initweight
            if good:
                condweight = condweight + initweight
        return  (condweight/totalweight)


class LikelihoodWeighting:

    cloudyday = BayesNet()
    resarray = []
    evidence = {}
    outcome = {}

    def __init__(self):
        # Init nodes
        cloudy = self.cloudyday.addnode("cloudy")
        sprinkler = self.cloudyday.addnode("sprinkler")
        rain = self.cloudyday.addnode("rain")
        wetgrass = self.cloudyday.addnode("wetgrass")

        # Connect nodes
        self.cloudyday.connection(cloudy.name, sprinkler.name)
        self.cloudyday.connection(cloudy.name, rain.name)
        self.cloudyday.connection(sprinkler.name, wetgrass.name)
        self.cloudyday.connection(rain.name, wetgrass.name)

        # Initialise empty arrays
        self.cloudyday.initprobsnew()

        # Initialise probabilities

        self.cloudyday.addprobs("cloudy", {'prob': 0.5})

        self.cloudyday.addprobs("sprinkler", {'cloudy': 1, 'prob': 0.1})
        self.cloudyday.addprobs("sprinkler", {'cloudy': 0, 'prob': 0.5})

        self.cloudyday.addprobs("rain", {'cloudy': 1, 'prob': 0.8})
        self.cloudyday.addprobs("rain", {'cloudy': 0, 'prob': 0.2})

        self.cloudyday.addprobs("wetgrass", {'sprinkler': 1, 'rain': 1, 'prob': 0.99})
        self.cloudyday.addprobs("wetgrass", {'sprinkler': 1, 'rain': 0, 'prob': 0.90})
        self.cloudyday.addprobs("wetgrass", {'sprinkler': 0, 'rain': 1, 'prob': 0.90})
        self.cloudyday.addprobs("wetgrass", {'sprinkler': 0, 'rain': 0, 'prob': 0.00})

        # Assign evidence and outcome according to assignment
        self.evidence = {"sprinkler": 1, "wetgrass": 1}
        self.outcome = {"cloudy": 1}

def main(argv):
    # Checking for correct input
    if len(argv) is not 3:
        print "Incorrect usage, 2 arguments required."
        return
    if not argv[1].isdigit() or not argv[2].isdigit():
        print "Incorrect usage, 2 integers required"
        return

    # Create variables
    numsamples = int(argv[1])
    numruns = int(argv[2])

    for x in xrange(0, numruns):
        LikelihoodWeighting.resarray.append(LikelihoodWeighting.cloudyday.likelihoodweighting(LikelihoodWeighting.evidence, LikelihoodWeighting.outcome, numsamples))

    mean = np.mean(LikelihoodWeighting.resarray)
    var = np.var(LikelihoodWeighting.resarray)

    print('%d %d' % (mean, var))

if __name__ == "__main__":
    main(sys.argv)