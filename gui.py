# importing a library pygame, for handling gui and opening a window for the user to interact with.
import pygame
from network import guess

# initialise pygame
pygame.init()

# global constants
WIDTH, HEIGHT = 280, 280
NEURONS = 784
FPS = 60
RADIUS = 1.5
BRUSH_RATE = 1
BACKGROUND_COLOR = (255, 255, 255)

# setting up a pygame window and its title.
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Digit Demo")

# initialising the matrix, which is going to store the values of each pixels.
matrix = []

# making the matrix a 28 x 28 zero matrix.
for y in range(HEIGHT // 10):
    matrix.append([])
    for x in range(WIDTH // 10):
        matrix[y].append(0)

# a function, to clear the matrix by making all the values zero
def clear():
    for i in range(WIDTH // 10):
        for j in range(HEIGHT // 10):
            matrix[i][j] = 0

# function, to handle when the user wants to submit a digit drawn.
# pushes the matrix into a txt file, which can be accessed by the neural network later.
def push():
    print(f"The digit drawn, is: {guess(matrix)}")
    clear()  # clearing the matrix after a user submits a digit drawn.

# drawing the 28 x 28 pixels with their appropriate grayscale colors.
def draw(window):
    # refreshing the background color after every frame.
    window.fill(BACKGROUND_COLOR)

    # handling mouse strokes.
    if pygame.mouse.get_pressed()[0]:
        mouse_x, mouse_y = pygame.mouse.get_pos()[0], pygame.mouse.get_pos()[1]
        pixel_x, pixel_y = mouse_x // 10, mouse_y // 10
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                nx, ny = pixel_x + dx, pixel_y + dy
                if 0 <= nx < WIDTH // 10 and 0 <= ny < HEIGHT // 10:
                    dist = (dx ** 2 + dy ** 2) ** 0.5
                    strength = max(0, 1 - dist / RADIUS)
                    matrix[ny][nx] = min(1, matrix[ny][nx] + strength * BRUSH_RATE)

    # drawing the pixels in their appropriate positions.
    for i, row in enumerate(matrix):
        for j, pixel in enumerate(row):
            pygame.draw.rect(window, (int(pixel * 255), int(pixel * 255), int(pixel * 255)), (j * 10, i * 10, 10, 10))

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
            
            # functionailty for clearing the drawing screen.
            if event.type == pygame.KEYDOWN and event.key == pygame.K_c:
                clear()

            # functionality for submitting a drawn digit, by pressing the 'enter' key.
            if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                push()
    
    # quiting the window after the main loop is exited
    pygame.quit()

if __name__ == '__main__':
    main()
