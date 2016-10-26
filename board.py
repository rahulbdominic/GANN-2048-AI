from Tkinter import *
from boardlogic import *
from random import *
import os

GRID_LEN = 4

class GameGrid(object):
    def __init__(self, matrix = []):
        self.grid_cells = []
        if(len(matrix) == 0):
            self.init_matrix()
        else:
            self.matrix = matrix

    def init_matrix(self):
        self.matrix = new_game(4)
        self.matrix=add_two(self.matrix)
        self.matrix=add_two(self.matrix)

    def move(self, event):
        print self.matrix
        if event == 'right':
            self.matrix, done, invalid = right(self.matrix)
        elif event == 'left':
            self.matrix, done, invalid = left(self.matrix)
        elif event == 'up':
            self.matrix, done, invalid = up(self.matrix)
        else:
            self.matrix, done, invalid = down(self.matrix)
        return (self.matrix, done, invalid)

    def key_down(self, event):
        key = repr(event.char)
        if key in self.commands:
            self.matrix,done = self.commands[repr(event.char)](self.matrix)
            if done:
                self.matrix = add_two(self.matrix)
                self.update_grid_cells()
                done=False
                if game_state(self.matrix)=='win':
                    self.grid_cells[1][1].configure(text="You",bg=BACKGROUND_COLOR_CELL_EMPTY)
                    self.grid_cells[1][2].configure(text="Win!",bg=BACKGROUND_COLOR_CELL_EMPTY)
                if game_state(self.matrix)=='lose':
                    self.grid_cells[1][1].configure(text="You",bg=BACKGROUND_COLOR_CELL_EMPTY)
                    self.grid_cells[1][2].configure(text="Lose!",bg=BACKGROUND_COLOR_CELL_EMPTY)

    def generate_next(self):
        index = (self.gen(), self.gen())
        while self.matrix[index[0]][index[1]] != 0:
            index = (self.gen(), self.gen())
        self.matrix[index[0]][index[1]] = 2

    def get_matrix(self):
        return self.matrix

    def print_matrix(self):
        os.system('clear')
        for item in self.matrix:
            print item
