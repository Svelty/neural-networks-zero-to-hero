import math
import numpy as np
import matplotlib.pyplot as plt

from graphviz import Digraph


def main():
    print("Hello test !")

    print(np.arange(-5,5,0.2))

    print(np.tanh(np.arange(-5,5,0.2)))

    plt.plot(np.arange(-5,5,0.2), np.tanh(np.arange(-5,5,0.2)))
    plt.grid()

    # plt.savefig("tanh.png") #save

    # plt.show()

    print("goodby")
    graph_vis_test()


def graph_vis_test():
    print('graphviz test')
    dot = Digraph(comment='Whats a digraph?')

    dot.node('a', 'AH')
    dot.node('b', 'ok')
    dot.node('c', 'i get it')

    dot.edges(['bc', 'ac'])
    dot.edge('b', 'a', constraint='false') #constraint = 'false' tells the dot engine not to stack these nodes

    print(dot.source)

    # dot.render('doctest-output/ok.gv').replace('\\', '/')
    # dot.render('doctest-output/ok', format='png', view=True)
    dot.view()

    print('graphviz test end')


main()



