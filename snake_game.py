# time to build the snake game
import pygame
import random
import numpy as np
from collections import namedtuple
from enum import Enum

# initialize pygame
pygame.init()
# see available fonts
# print(pygame.font.get_fonts())
# set font
pygame.font.init()
font = pygame.font.SysFont("comicsans", 30)
# apple = pygame.image.load('apple.png')
# pygame.transform.scale(apple, (20,20))

# stuff to add to normal game:
# reset function (agent redoes game)
# reward function
# change play(action) -> direction
# current game or gamenumber
# change died function

# numerical system for the direction of the snake
# enumeration = data type that defines a set of values
# making a class for new data types
class direct(Enum):
    right = 1
    left = 2
    up = 3
    down = 4

# define a point in the environment
# namedtuple is like a list but immutable

pt = namedtuple('pt', 'x, y')

# making the size of the obj/snake

size = 20

# and speed

# faster = train faster
speed = 99999999999999999999999

# some colors

white = (255,255,255)
black = (0,0,0)
red = (190, 0, 0)
red2 = (255, 0, 0)
blue = (0,200, 0)
yellow = (0,240,0)

# class for the actual game
class runAI:

    def __init__(self, l = 600, w=600):
        self.l=l
        self.w=w

        # make the display

        self.display = pygame.display.set_mode((self.l, self.w))
        pygame.display.set_caption('Snake Robot')
        self.clock = pygame.time.Clock()



        self.reset()



    def reset(self):
        # make the starting conditions of the pygame
        self.direct = direct.right
        self.front = pt(self.l/2, self.w/2)
        self.body = [self.front, pt(self.front.x-size, self.front.y),pt(self.front.x-(2*size),self.front.y)]

        self.score=0
        self.obj= None
        self.initobj()
        self.framenumber = 0

    # pick a random point in the environment to put an object
    # later, make it so that collecting object gives score
    def initobj(self):
        x = random.randint(0, (self.l-size)//size)*size
        y = random.randint(0, (self.w-size)//size)*size
        self.obj = pt(x, y)
        # apple.get_rect().centerx = self.obj.x
        # apple.get_rect().centery = self.obj.y
        # if obj is gone, make another obj
        if self.obj in self.body:
            self.initobj()
        # get size of image
        # width, height = apple.get_size()
        # print size of image
        # print('Width:', width, 'Height:', height)



    def play(self, action):
        self.framenumber += 1
        # seeing what user does
        # making it so stuff moves where user wants
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()




            # if event.type == pygame.KEYDOWN:
               # if event.key == pygame.K_LEFT:
                #    self.dir = direct.left
               # elif event.key == pygame.K_RIGHT:
                #    self.dir = direct.right
                #elif event.key == pygame.K_UP:
                 #   self.dir = direct.up
               # elif event.key == pygame.K_DOWN:
                   # self.dir = direct.down

        # actually moving
        self.move(action)
        self.body.insert(0,self.front)

        reward = 0
        finish = False
        if self.died() or self.framenumber > 100*len(self.body):
            finish = True
            reward = -10
            return reward, finish, self.score

        if self.front == self.obj:
            self.score += 5
            reward = 10
            self.initobj()
        else:
            self.body.pop()

        self.changeinterface()
        self.clock.tick(speed)

        return reward, finish, self.score

    def died(self, pt=None):
        if pt is None:
            pt = self.front
        # if the snake goes into itself
        if pt in self.body[1:]:
            return True
        # if the snake hits the edge
        if pt.x > self.l - size or pt.x < 0 or pt.y > self.w - size or pt.y < 0:
            return True


        return False

    def changeinterface(self):
        self.display.fill(black)
        for pt in self.body:
            pygame.draw.rect(self.display, yellow, pygame.Rect(pt.x, pt.y, size, size))
            pygame.draw.rect(self.display, blue, pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        # self.display.blit(apple,[self.obj.x,self.obj.y])
        # for pt in self.front:
            # pygame.draw.rect(self.display, red, pygame.Rect(self.obj.x, self.obj.y, size, size))
            # pygame.draw.rect(self.display, red2, pygame.Rect(self.obj.x+4, self.obj.y+4, 12, 12))

        pygame.draw.rect(self.display, red, pygame.Rect(self.obj.x, self.obj.y, size, size))

        txt = font.render("score: " + str(self.score), True, white)
        self.display.blit(txt,[10,10])
        pygame.display.flip()

    def move(self, action):
        # [straight, right, left]
        clock_wise = [direct.right, direct.down, direct.left, direct.up]
        windex = clock_wise.index(self.direct)
        if np.array_equal(action, [1,0,0]):
            newdir = clock_wise[windex] # no change
        elif np.array_equal(action, [0,1,0]):
            nextwindex = (windex+1)%4
            newdir = clock_wise[nextwindex] # right turn clockwise
        else: # [0,0,1]
            nextwindex = (windex-1)%4
            newdir = clock_wise[nextwindex] # left turn clockwise

        self.direct = newdir

        x = self.front.x
        y = self.front.y
        if self.direct == direct.right:
            x += size
        elif self.direct == direct.left:
            x -= size
        elif self.direct == direct.up:
            y -= size
        elif self.direct == direct.down:
            y += size
        self.front = pt(x,y)
        # print('front:' , self.front)
        # print('obj:' , self.obj)
