import random
import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
## THIS CODE IS AN IMPLEMENTATION OF A TETRIS VERSION IN PYTHON
## THIS CODE WAS NOT MADE BY ME
## CREDITS: https://levelup.gitconnected.com/writing-tetris-in-python-2a16bddb5318
## IT HAS BEEN MODIFIED TO MATCH THE GBA VERSION: TETRIS WORLD I'M USING AS A REAL ENVIRONMENT. THIS IS SO THE SIMULATED ENVIRONMENT IS AS CLOSE AS POSSIBLE TO THE REAL GBA GAME.


colors = [
    (0, 0, 0),
    (120, 37, 179),
    (100, 179, 179),
    (80, 34, 22),
    (80, 134, 22),
    (180, 34, 22),
    (180, 34, 122),
]


class Figure:
    x = 0
    y = 0
    figures = [
        [[4, 5, 6, 7], [1, 5, 9, 13]],
        [[4, 5, 9, 10], [2, 6, 5, 9]],
        [[6, 7, 9, 10], [1, 5, 6, 10]],
        [[0, 4, 5, 6], [1, 5, 9, 8], [4, 5, 6, 10], [1, 2, 5, 9]],
        [[3, 5, 6, 7], [1, 2, 6, 10], [5, 6, 7, 9], [2, 6, 10, 11]],
        [[1, 4, 5, 6], [1, 4, 5, 9], [4, 5, 6, 9], [1, 5, 6, 9]],
        [[1, 2, 5, 6]],
    ]
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.type = random.randint(0, len(self.figures) - 1)
        self.color = random.randint(1, len(colors) - 1)
        self.rotation = 0

    def image(self):
        return self.figures[self.type][self.rotation]

    def rotate(self):
        self.rotation = (self.rotation + 1) % len(self.figures[self.type])


class Tetris:
    level = 8
    score = 0
    state = "start"
    field = []
    height = 0
    width = 0
    x = 100
    y = 60
    zoom = 20
    figure = None

    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.field = []
        self.score = 0
        self.state = "start"
        for i in range(height):
            new_line = []
            for j in range(width):
                new_line.append(0)
            self.field.append(new_line)

    def new_figure(self):
        self.figure = Figure(3, 0)

    def intersects(self):
        intersection = False
        for i in range(4):
            for j in range(4):
                if i * 4 + j in self.figure.image():
                    if i + self.figure.y > self.height - 1 or \
                            j + self.figure.x > self.width - 1 or \
                            j + self.figure.x < 0 or \
                            self.field[i + self.figure.y][j + self.figure.x] > 0:
                        intersection = True
        return intersection

    def break_lines(self):
        lines = 0
        for i in range(1, self.height):
            zeros = 0
            for j in range(self.width):
                if self.field[i][j] == 0:
                    zeros += 1
            if zeros == 0:
                lines += 1
                for i1 in range(i, 1, -1):
                    for j in range(self.width):
                        self.field[i1][j] = self.field[i1 - 1][j]
        self.score += lines ** 2

    def go_down(self):
        self.figure.y += 1
        if self.intersects():
            self.figure.y -= 1
            self.freeze()

    def freeze(self):
        for i in range(4):
            for j in range(4):
                if i * 4 + j in self.figure.image():
                    self.field[i + self.figure.y][j + self.figure.x] = self.figure.color
        self.break_lines()
        self.new_figure()
        if self.intersects():
            self.state = "gameover"

    def go_side(self, dx):
        old_x = self.figure.x
        self.figure.x += dx
        if self.intersects():
            self.figure.x = old_x

    def rotate(self):
        old_rotation = self.figure.rotation
        self.figure.rotate()
        if self.intersects():
            self.figure.rotation = old_rotation


class Simulation:
    def __init__(self):
        self.done = False

        self.game = Tetris(20, 10)

        self.pressing_down = False

        self.alternate = True
        self.game.__init__(20, 10)
        self.previous_score = 0.0

    def reset(self):
        self.done = False
        self.game.__init__(20,10)
        return self.get_observation()

    def get_observation(self):
        observation = -F.threshold(-torch.tensor(self.game.field, dtype=torch.float), -0.1, -1.0).unsqueeze(0)
        if self.game.figure is not None:
            for i in range(4):
                for j in range(4):
                    p = i * 4 + j
                    if p in self.game.figure.image():
                        observation[0, i + self.game.figure.y, j + self.game.figure.x] = 1.0
        return observation.to(device)

    def run_step(self, action):
        self.previous_score = self.game.score

        if self.game.figure is None:
            self.game.new_figure()

        if action == 1:
            self.game.rotate()
        elif action == 4:
            self.pressing_down = True
        if action == 2:
            self.game.go_side(-1)
        if action == 3:
            self.game.go_side(1)

        if self.game.state == "start":
            if self.pressing_down:
                for _ in range(4):
                    self.game.go_down()
            elif self.alternate:
                self.game.go_down()
            else:
                self.game.go_down()
                self.game.go_down()
            self.alternate = not self.alternate
        self.pressing_down = False

        observation = self.get_observation()
        self.done = self.game.state == "gameover"
        if self.done:
            reward = torch.tensor(-1.0, dtype=torch.float).to(device)
        else:
            reward = torch.tensor((self.game.score - self.previous_score)/16, dtype=torch.float).to(device)

        return observation, self.done, reward
