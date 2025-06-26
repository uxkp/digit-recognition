# importing a library pygame, for handling gui and opening a window for the user to interact with.
import random
import pygame

# initialise pygame
pygame.init()

# global constants
WIDTH, HEIGHT = 280, 280
FPS = 60
BACKGROUND_COLOR = (255, 255, 255)

def reshape_to_matrix(flat_list):
    assert len(flat_list) == 784, "Input must be a list of 784 values"
    return [flat_list[i*28:(i+1)*28] for i in range(28)]

# setting up a pygame window and its title.
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Digit Demo")

# loading the mnist data.
def load_data(filename):
    data = []
    with open(filename, "r") as f:
        next(f)
        for line in f:
            parts = line.strip().split(",")
            pixels = [int(x) / 255.0 for x in parts[1:]]  # normalize
            data.append(pixels)
    return data

# initialising the matrix, which is going to store the values of each pixels.
training_data = load_data("./data/mnist_train.csv")[:100]
matrix = reshape_to_matrix(training_data[random.randint(1, 100)])

# drawing the 28 x 28 pixels with their appropriate grayscale colors.
def draw(window):
    # refreshing the background color after every frame.
    window.fill(BACKGROUND_COLOR)

    # drawing the pixels in their appropriate positions.
    for i, row in enumerate(matrix):
        for j, pixel in enumerate(row):
            pygame.draw.rect(window, (int(pixel * 255), int(pixel * 255), int(pixel * 255)), (j * 10, i * 10, 10, 10))

    # drawing a 20 x 20 square, where the digit should be ideally centered.
    pygame.draw.line(window, (15, 15, 15), (40, 40), (240, 40))
    pygame.draw.line(window, (15, 15, 15), (40, 40), (40, 240))
    pygame.draw.line(window, (15, 15, 15), (240, 40), (240, 240))
    pygame.draw.line(window, (15, 15, 15), (40, 240), (240, 240))

    pygame.display.update()

# main function, which renders everything onto the window, consisting of the main loop which refreshes at 60fps.
def main():
    run = True

    # setting up a pygame clock, so that high performance computers have a minimum refresh rate of 60fps.
    clock = pygame.time.Clock()

    # main loop
    while run:
        clock.tick(FPS)
        draw(WIN)

        # handling if the user quits the window.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                break
    
    # quiting the window after the main loop is exited
    pygame.quit()

if __name__ == '__main__':
    main()
