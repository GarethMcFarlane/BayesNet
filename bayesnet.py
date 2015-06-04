__authors__ = 'Yoshi Hemzal, Gareth McFarlane'

import sys
import itertools
import copy
import random
import numpy as np


# Definition of the Node class, with name, probabilities, parent and child variables.
class Node:
    def __init__(self, name):
        self.name = name
        self.probabilities = None
        self.parent = []
        self.child = []

    # Add a child to the node
    def addchild(self, c):
        self.child.append(c)

    # Add a parent to the node
    def addparnt(self, p):
        self.parent.append(p)

    # Remove a child from a node by name
    def delchild(self, name):
        for c in self.child:
            if c.name is name:
                self.child.remove(c)
                return True
        return False

    # Remove a parent in a node by name
    def delparnt(self, name):
        for p in self.parent:
            if p.name is name:
                self.parent.remove(p)
                return True
        return False

    # Initialise probability tables to have a size relative to the number of parents a node has
    def initprobs(self):
        totalparents = len(self.parent)
        if totalparents is 0:
            self.probabilities = {'prob': None}
            return True

        totalcombinations = map(''.join, itertools.product("10", repeat=totalparents))
        self.probabilities = {}
        for t in totalcombinations:
            self.probabilities[t] = None
        return True

    # Check if probabilities have been initialised. If not, return false.
    def checkprobsinit(self):
        for id in self.probabilities:
            if self.probabilities[id] is None:
                return False
        return True

    # Set conditional probabilities for a node
    def setcondprobs(self, probs):
        if 'prob' not in probs:
            return False
        if not self.parent:
            self.probabilities = {'prob': float(probs['prob'])}
            return True

        id = ""
        for p in self.parent:
            id += str(probs[p.name])
        self.probabilities[id] = float(probs['prob'])
        return True

# Definition of the Bayesian Network class, with Nodes and Probability variables
class BayesNet:
    def __init__(self):
        self.nodes = []
        self.probs = []

    # Make an exact copy of a network for sorting purposes
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

    # Topological sort of a network. Sorting ensures requirements for calculated weights are met i.e. no errors in
    # probabilities.
    def topsort(self):
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
        if net.checkconnections() is True:
            return False
        for node in sorted:
            final.append(self.getnode(node.name))
        return final

    # Add a node to the network, by name.
    def addnode(self, nodename):
        if self.checkexists(nodename) is False:
            scatchnode = Node(nodename)
            self.nodes.append(scatchnode)
            return scatchnode
        return False

    # Returns a node matching the name requested (if no match, return false).
    def getnode(self, nodename):
        for i in self.nodes:
            if i.name is nodename:
                return i
        return False

    # Check if a node already exists to avoid duplicating
    def checkexists(self, name):
        for n in self.nodes:
            if n.name is name:
                return True
        return False

    # Gets the root node of a network
    def getrootnode(self):
        rootarr = []
        for n in self.nodes:
            if not n.parent:
                rootarr.append(n)
        return rootarr

    # Checks if a node is a parent or child. Any node in a Bayesian network must have at least on of these.
    def checkconnections(self):
        for n in self.nodes:
            if n.parent or n.child:
                return True
        return False

    # Connects two nodes (specified by parent and child)
    def connection(self, parentnode, childnode):
        parent = self.getnode(parentnode)
        child = self.getnode(childnode)

        if (parent, child) is not False:
            parent.addchild(child)
            child.addparnt(parent)
            return True
        return False

    # Disconnects two nodes (specified by parent and child)
    def disconnection(self, parentnode, childnode):
        parent = self.getnode(parentnode)
        child = self.getnode(childnode)

        if (parent, child) is not False:
            parent.delchild(childnode)
            child.delparnt(parentnode)
            return True
        return False

    # Initialises empty probabilities for a new node.
    def initprobsnew(self):
        for node in self.nodes:
            node.initprobs()
        return True

    # Checks if probabilities have been successfully initialised.
    def checkprobsinit(self):
        for node in self.nodes:
            if not node.checkprobsinit():
                return False
        return True

    # Adds probabilities to a requested node.
    def addprobs(self, nodename, probs):
        return self.getnode(nodename).setcondprobs(probs)

    # Calculates sample weight. Initial weight is 1. Multiplies weight based on node probabilities. Uses random.random
    # to determine the outcome of nodes that are not fixed. For every node, we check if the outcome passes the
    # requisites of the fixed nodes. Basic sorting allows for correct probability weighting. This represents one run of
    # the "sample".

    def getweight(self, nodes, evidence, outcome):
        resarray = {}
        initweight = 1.0
        good = True

        for n in nodes:
            if len(n.probabilities) is 1:
                if n.name in evidence:
                    if evidence[n.name] is 1:
                        resarray[n.name] = 1
                        initweight *= n.probabilities['prob']
                    else:
                        resarray[n.name] = 0
                        initweight *= 1.0 - n.probabilities['prob']

                else:
                    if n.probabilities['prob'] >= random.random():
                        resarray[n.name] = 1
                    else:
                        resarray[n.name] = 0
            else:
                id = ""
                for p in n.parent:
                    id += str(resarray[p.name])
                if n.name in evidence:
                    if evidence[n.name] is 1:
                        resarray[n.name] = 1
                        initweight *= n.probabilities[id]
                    else:
                        resarray[n.name] = 0
                        initweight *= 1 - n.probabilities[id]
                else:
                    if n.probabilities[id] >= random.random():
                        resarray[n.name] = 1
                    else:
                        resarray[n.name] = 0

            if good and (n.name in outcome):
                if outcome[n.name] != resarray[n.name]:
                    good = False

        return good, initweight

    # Likelihood weighting, run as many times as is requested.
    def likelihoodweighting(self, evidence, outcome, samplenum):
        if not self.checkprobsinit():
            return False

        sorted = self.topsort()

        if sorted is False:
            return False

        # Totalweight, initially set to 0. Conditional weight is set to a very small initial number to avoid
        # later errors in the main method (division by zero).
        totalweight = 0.0
        condweight = 0.0001

        # Run for as many samples are requested.
        for x in xrange(0, samplenum):
            good, initweight = self.getweight(sorted, evidence, outcome)
            totalweight += initweight
            if good:
                condweight += initweight
        return condweight / totalweight

# Main method using the classes above. Takes 2 arguments (integers) and returns
# mean and variance of the posterior estimate for m samples and n runs.

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

    # Create array with resulting probabilities
    results = []

    # Run the network for as many runs as specified, with as many samples as specified
    for x in xrange(0, numruns):
        results.append(cloudyday.likelihoodweighting(evidence, outcome, numsamples))

    # Use numpy to quickly get results
    mean = np.mean(results)
    variance = np.var(results)

    # Print results
    print('%f %f' % (mean, variance))

if __name__ == "__main__":
    main(sys.argv)
