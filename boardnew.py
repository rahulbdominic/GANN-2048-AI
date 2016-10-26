from __future__ import division
import random

class GameGrid(object):
    def __init__(self, matrix = []):
        self.map_length = 4
        self.end_number = 32768
        #initialize array
        if len(matrix) == 0:
            self.arr = [[0 for x in range(self.map_length)] for x in range(self.map_length)]
        else:
            self.arr = matrix
        self.__generate_random(3)

    def __generate_random(self, max):
        counter = 0
        while (counter < max):
            row = random.randint(0, self.map_length - 1)
            col = random.randint(0, self.map_length - 1)
            if self.arr[row][col] == 0:
                self.arr[row][col] = int(random.choice("24"))
                counter += 1

    def end_turn(self):
        can_move = False
        for i in range(0, self.map_length):
            for j in range(0, self.map_length):
                if self.arr[i][j] == 0:
                    can_move = True
                    break
                    if can_move:
                        self.__generate_random(1)
                    else:
                        print "Game Over"
        return can_move

    def max_matrix(self):
        maxed = max([max(sub_array) for sub_array in self.arr])
        return maxed

    def scalar_multiply(self):
        arr_new = [[self.arr[x][y] for y in range(len(self.arr[0]))] for x in range(len(self.arr))]
        maxed = self.max_matrix()
        for i in range(4):
            for j in range(4):
                arr_new[i][j] = arr_new[i][j] / maxed
        return arr_new

    def get_matrix(self):
        return self.arr

    def print_result(self):
        print("\n".join(["".join(["{:5}".format(item) for item in row]) for row in self.arr]))

    def move(self, dir):
        has_moved = False
        invalid = False

        if dir == "w":
            for col in range(0, self.map_length):
                for row_to_be_filled in range(0, self.map_length):
                    for row in range(row_to_be_filled, self.map_length):
                        if self.arr[row][col] != 0:
                            # select this one, check if can combine
                            is_double = False
                            for row_comb in range(row+1, self.map_length):
                                if self.arr[row_comb][col] != 0:
                                    if self.arr[row_comb][col] == self.arr[row][col]:
                                        self.arr[row_to_be_filled][col] = self.arr[row][col] * 2
                                        self.arr[row_comb][col] = 0
                                        if row_to_be_filled != row:
                                            self.arr[row][col] = 0
                                        is_double = True
                                        has_moved = True
                                        break
                                    else:
                                        break
                            # no valid number in other rows been found
                            if not is_double:
                                self.arr[row_to_be_filled][col] = self.arr[row][col]
                                if row_to_be_filled != row:
                                    self.arr[row][col] = 0
                                    has_moved = True
                            # check if ended
                            elif self.arr[row_to_be_filled][col] == self.end_number:
                                print "You win!!!"
                                exit(0)
                            break

        elif dir == "s":
            for col in range(self.map_length-1, -1, -1):
                for row_to_be_filled in range(self.map_length-1, -1, -1):
                    for row in range(row_to_be_filled, -1, -1):
                        if self.arr[row][col] != 0:
                            # select this one, check if can combine
                            is_double = False
                            for row_comb in range(row-1, -1, -1):
                                if self.arr[row_comb][col] != 0:
                                    if self.arr[row_comb][col] == self.arr[row][col]:
                                        self.arr[row_to_be_filled][col] = self.arr[row][col] * 2
                                        self.arr[row_comb][col] = 0
                                        if row_to_be_filled != row:
                                            self.arr[row][col] = 0
                                        is_double = True
                                        has_moved = True
                                        break
                                    else:
                                        break
                            # no valid number in other rows been found
                            if not is_double:
                                self.arr[row_to_be_filled][col] = self.arr[row][col]
                                if row_to_be_filled != row:
                                    self.arr[row][col] = 0
                                    has_moved = True
                            # check if ended
                            elif self.arr[row_to_be_filled][col] == self.end_number:
                                print "You win!!!"
                                exit(0)
                            break

        elif dir == "a":
            for row in range(0, self.map_length):
                for col_to_be_filled in range(0, self.map_length):
                    for col in range(col_to_be_filled, self.map_length):
                        if self.arr[row][col] != 0:
                            # select this one, check if can combine
                            is_double = False
                            for col_comb in range(col+1, self.map_length):
                                if self.arr[row][col_comb] != 0:
                                    if self.arr[row][col_comb] == self.arr[row][col]:
                                        self.arr[row][col_to_be_filled] = self.arr[row][col] * 2
                                        self.arr[row][col_comb] = 0
                                        if col_to_be_filled != col:
                                            self.arr[row][col] = 0
                                        is_double = True
                                        has_moved = True
                                        break
                                    else:
                                        break
                            # no valid number in other rows been found
                            if not is_double:
                                self.arr[row][col_to_be_filled] = self.arr[row][col]
                                if col_to_be_filled != col:
                                    self.arr[row][col] = 0
                                    has_moved = True
                            # check if ended
                            elif self.arr[row][col_to_be_filled] == self.end_number:
                                print "You win!!!"
                                exit(0)
                            break

        elif dir == "d":
            for row in range(self.map_length-1, -1, -1):
                for col_to_be_filled in range(self.map_length-1, -1, -1):
                    for col in range(col_to_be_filled, -1, -1):
                        if self.arr[row][col] != 0:
                            # select this one, check if can combine
                            is_double = False
                            for col_comb in range(col-1, -1, -1):
                                if self.arr[row][col_comb] != 0:
                                    if self.arr[row][col_comb] == self.arr[row][col]:
                                        self.arr[row][col_to_be_filled] = self.arr[row][col] * 2
                                        self.arr[row][col_comb] = 0
                                        if col_to_be_filled != col:
                                            self.arr[row][col] = 0
                                        is_double = True
                                        has_moved = True
                                        break
                                    else:
                                        break
                            # no valid number in other rows been found
                            if not is_double:
                                self.arr[row][col_to_be_filled] = self.arr[row][col]
                                if col_to_be_filled != col:
                                    self.arr[row][col] = 0
                                    has_moved = True
                            # check if ended
                            elif self.arr[row][col_to_be_filled] == self.end_number:
                                print "You win!!!"
                                exit(0)
                            break

        else:
            print("invalid command")

        done = False
        if has_moved:
            can_move = self.end_turn()
            if can_move == False:
                done = True
        else:
            invalid = True

        return (self.arr, done, invalid)
# end class Grid
