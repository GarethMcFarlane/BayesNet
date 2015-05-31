__authors__ = 'Gareth McFarlane, Yoshi Hemzal'

import sys
import numpy

class LikelihoodWeighting:

    # TODO: Init function

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

        # Initialise network
    # TODO: Incorporate in init.


    if __name__ =="__main__":
        main(sys.argv)


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

    def getchildren(self):
        return self.child

    def getparents(self):
        return self.parent

class BayesNet:
    def __init__(self):
        self.nodes = []
        self.probs = []

    def addnode(self, nodename):
        self.nodes.append(Node(nodename))

    def getnode(self, nodename):
        for i in self.nodes:
            if i.name is nodename:
                return i
        return False

    def connection(self, parentnode, childnode):
        parent = self.getnode(parentnode)
        child = self.getnode(childnode)

        if (parent, child) is not False:
            parent.addchild(child)
            child.addparnt(parent)
            return True
        return False