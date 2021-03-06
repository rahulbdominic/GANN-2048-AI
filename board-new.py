import random

class GameGrid(object):

    # Todo:
    # Integrate new grid class into evolution.py
    def __init__(self, matrix = []):
        self.map_length = 4
        self.end_number = 32768
        #initialize array
        if len(matrix) == 0:
            self.arr = [[0 for x in range(self.map_length)] for x in range(self.map_length)]
        else:
            self.arr = matrix
        self.generate_random(3)

    def generate_random(self, max):
        ounter = 0
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

    def get_matrix():
        return self.arr

    def print_result(self):
	print("\n".join(["".join(["{:5}".format(item) for item in row]) for row in self.arr]))

    def move(self, dir):
	has_moved = False
        invalid = False

	if dir == "w":
            print("up")
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
					break

	elif dir == "s":
	    print("down")
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
			print("left")
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
			print("right")
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

		if has_moved:
			self.end_turn()
            if can_move:
                done = True
        else:
            invalid = True
        return (self.arr, done, invalid)
# end class Grid

# start game
game_map = GameGrid()
# each iteration
while True:
	game_map.print_result()
	cmd = raw_input("please enter command w / a / s / d to move, q to quit\n > ")
	if cmd == "q":
		exit(0)
	elif cmd in "wasd":
		game_map.move(cmd)
	else:
		print("invalid command")
