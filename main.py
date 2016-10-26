from network import Network
from evolution import *
import numpy as np
from boardnew import *


'''
NN Basic Test:
constructor: size seed
net = Network([3, 4, 1], 1)
net.evaluate(np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]]))
'''

'''
2048 Basic Test:
gamegrid = GameGrid()
gamegrid.print_matrix()
gamegrid.move('up')
gamegrid.print_matrix()
gamegrid.move('up')
gamegrid.print_matrix()
'''


net = Network([16, 24, 4], 10)

gamegrid = GameGrid()
game_matrix = gamegrid.get_matrix()

evonet = EvolveNet(net, game_matrix)
g_algo = GeneticAlgorithm(evonet)
g_algo.run()
