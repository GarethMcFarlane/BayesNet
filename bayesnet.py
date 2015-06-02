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

    ### NEW
    def checkchild(self, checkname):
        for c in self.child:
            if c.name == checkname:
                return True
        return False

    def checkparnt(self, checkname):
        for p in self.parent:
            if p.name == checkname:
                return True
        return False


    ### END NEW

    def addchild(self, c):
        # NEW
        if self.checkchild(c.name) is False:
            self.child.append(c)
            return True
        return False


    def addparnt(self, p):
        # NEW
        if self.checkparnt(p.name) is False:
            self.parent.append(p)
            return True
        return False

    def remchild(self, name):
        for c in self.child:
            if c.name == name:
                self.child.remove(c)
                return True
        return False

    def remparnt(self, name):
        for p in self.parent:
            if p.name == name:
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
        if 'prob' not in probs:
            return False
        if not self.parent:
            self.probabilities = {'prob':float(probs['prob'])}
            return True

        id = ""
        #NEW
        try:
            for p in self.parent:
                id += str(probs[p.name])
        except Exception:
            return False

        self.probabilities[id] = float(probs['prob'])
        return True

    #NEW
    def checkallprobsinit(self):
        for id in self.probabilities:
            if self.probabilities[id] == None:
                return False
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
            for c in j.child:
                net.connection(j.name, c.name)
        for k in self.nodes:
            net.getnode(k.name).probabilities = copy.deepcopy(k.probabilities)

        return net

    def topsort(self):
        #NEW
        sorted = []
        final = []
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
        if (net.checkconnections() == True):
            return False
        for node in sorted:
            final.append(self.getnode(node.name))
        return final

    # NEW
    def checkcycles(self):
        if (self.topsort() == False):
            return True
        return False

    def addnode(self, nodename):
        if self.checkexists(nodename) is False:
            scatchnode = Node(nodename)
            self.nodes.append(scatchnode)
            return scatchnode
        return False

    def getnode(self, nodename):
        for i in self.nodes:
            if i.name is nodename:
                return i
        return False

    #NEW
    def checkexists(self,name):
        for n in self.nodes:
            if n.name == name:
                return True
        return False

    def getallnodes(self):
        return self.nodes

    def getrootnode(self):
        rootarr = []
        for n in self.nodes:
            if not n.parent:
                rootarr.append(n)
        return rootarr

    #NEW
    def checkconnections(self):
        for n in self.nodes:
            if n.parent or n.child:
                return True
        return False

    def connection(self, parentnode, childnode):
        parent = self.getnode(parentnode)
        child = self.getnode(childnode)

        if (parent and child):
            parent.addchild(child)
            child.addparnt(parent)
            return True
        return False

    def disconnection(self, parentnode, childnode):
        parent = self.getnode(parentnode)
        child = self.getnode(childnode)

        if (parent and child):
            parent.remchild(childnode)
            child.remparnt(parentnode)
            return True
        return False

    def initprobsnew(self):
        for node in self.nodes:
            node.initprobs()
        return True

    def addprobs(self, nodename, probs):
        return self.getnode(nodename).setcondprobs(probs)

    #NEW
    def checkprobsinit(self):
        for node in self.nodes:
            if not node.checkallprobsinit():
                return False
        return True

    def getweight(self, nodes, evidence, outcome):
        resarray = {}
        initweight = 1.0
        good = True

        for n in nodes:
            if len(n.probabilities) == 1:
                if n.name in evidence:
                    if evidence[n.name] == 1:
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
            else:
                id = ""
                for p in n.parent:
                    id = id+str(resarray[p.name])
                if n.name in evidence:
                    if evidence[n.name] == 1:
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

            if ((good == True) and (n.name in outcome)):
                if (outcome[n.name] != resarray[n.name]):
                    good = False

        return good, initweight

    def likelihoodweighting(self, evidence, outcome, samplenum):
        if not self.checkprobsinit():
            return False

        sorted = self.topsort()

        if (sorted == False):
            return False

        totalweight = 0.0
        condweight = 0.0001
        for x in xrange(0, samplenum):
            good, initweight = self.getweight(sorted, evidence, outcome)
            totalweight += initweight
            if good:
                condweight = condweight + initweight
        return (condweight/totalweight)


def run_likelihood(network,evidence,outcome,num,numruns):
    results = []
    for x in xrange(0,numruns):
        results.append(network.likelihoodweighting(evidence, outcome, num))
    mean = np.mean(results)
    variance = np.var(results)
    print('%f %f' % (mean, variance))


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

    cloudyday = BayesNet()

    # Init nodes
    cloudy = cloudyday.addnode("cloudy")
    sprinkler = cloudyday.addnode("sprinkler")
    rain = cloudyday.addnode("rain")
    wetgrass = cloudyday.addnode("wetgrass")

    # Connect nodes
    cloudyday.connection(cloudy.name, sprinkler.name)
    cloudyday.connection(cloudy.name, rain.name)
    cloudyday.connection(sprinkler.name, wetgrass.name)
    cloudyday.connection(rain.name, wetgrass.name)

    # Initialise empty arrays
    cloudyday.initprobsnew()

    # Initialise probabilities

    cloudyday.addprobs("cloudy", {'prob': 0.5})

    cloudyday.addprobs("sprinkler", {'cloudy': 1, 'prob': 0.1})
    cloudyday.addprobs("sprinkler", {'cloudy': 0, 'prob': 0.5})

    cloudyday.addprobs("rain", {'cloudy': 1, 'prob': 0.8})
    cloudyday.addprobs("rain", {'cloudy': 0, 'prob': 0.2})

    cloudyday.addprobs("wetgrass", {'sprinkler': 1, 'rain': 1, 'prob': 0.99})
    cloudyday.addprobs("wetgrass", {'sprinkler': 1, 'rain': 0, 'prob': 0.90})
    cloudyday.addprobs("wetgrass", {'sprinkler': 0, 'rain': 1, 'prob': 0.90})
    cloudyday.addprobs("wetgrass", {'sprinkler': 0, 'rain': 0, 'prob': 0.00})

    # Assign evidence and outcome according to assignment
    evidence = {"sprinkler": 1, "wetgrass": 1}
    outcome = {"cloudy": 1}

    run_likelihood(cloudyday, evidence, outcome, numsamples, numruns)



if __name__ == "__main__":
    main(sys.argv)