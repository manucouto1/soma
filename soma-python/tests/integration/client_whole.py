"""The same graph undistributed, for comparison. It starts no process."""

import os

from net import graph, nodes

from somatize import Graph

print("OUTPUT", graph(Graph, nodes(), distributed=False).forward("  The Dog Runs Quickly  "))
print("HERE", float(os.getpid()))
