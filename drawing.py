import pygame
import numpy as np
from pygame.locals import *
import tensorflow as tf
from tensorflow import keras

model = keras.models.load_model("saved_model")


class board:
    def __init__(self):
        self.board = np.zeros((28,28))

    def setup_board(self, win):
        win.fill(white)
    def draw_tile(self, win, x, y):
        #going to want to add that it covers a radius of the tile clicked, or it will be tedious drawing numbers
        for i in range(-1,2):
            for j in range(-1,2):
                color = (0,0,0)
                val = 1
                pygame.draw.rect(win, color, ((x+i)*10, (y+j)*10, 10, 10))
                self.board[x+i,y+j] = val
        #print(self.board)
    def pen(self, win, x, y):
        draw_tile(self, win, x, y)
        draw_tile(self, win, x+1, y)
        draw_tile(self, win, x-1, y)
        draw_tile(self, win, x, y+1)
        draw_tile(self, win, x, y-1)

white = (255, 255, 255)
black = (0,0,0)

pygame.init()
win = pygame.display.set_mode((280, 280))
b = board()
win.fill(white)
drawing = True
doodle = False
while drawing:
    pygame.display.update()
    #pygame.init()
    raw_x, raw_y = pygame.mouse.get_pos()

    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            drawing = False
            #sys.exit()
        if event.type == KEYDOWN:
            if event.key == K_ESCAPE:
                pygame.quit()
                drawing = False
                #sys.exit()
        if event.type == MOUSEBUTTONDOWN:
            if event.button == 1:
                doodle = True

        if event.type == MOUSEBUTTONUP:
            if event.button == 1:
                doodle = False

        if doodle:
            x, y = raw_x//10, raw_y//10
            b.draw_tile(win, x, y)

pic = b.board
print(pic.shape)
arr = []
arr.append(pic)
arr = np.array(arr)
print(arr.shape)


prediction = model.predict(arr)
#print(prediction)
print(np.argmax(prediction[0]))
