import os
import json
import numpy as np
from tqdm import tqdm
import itertools

from random import sample
import random
from scipy.ndimage import affine_transform

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize

fun_data_dir = r'C:\Users\mmerl\projects\ARC-AGI\data_fun'


def plot_task(task, train_or_test="train"):
    """ plots a task with separating lines between subplots """
    examples = task[train_or_test]

    n_examples = len(examples)
    cmap = ListedColormap([
        '#000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
        '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'
    ])
    norm = Normalize(vmin=0, vmax=9)
    
    # Adjust the figure size
# Adjust the figure size and create 2 columns (for input and output) and n_examples rows
    figure, axes = plt.subplots(nrows=n_examples, ncols=2, figsize=(8, n_examples * 4))
    
    if len(examples)>1:
        # Plot the images: each example has its input in the left column and output in the right column
        for row, example in enumerate(examples):
            axes[row, 0].imshow(example['input'], cmap=cmap, norm=norm)   # Input in the first column
            axes[row, 1].imshow(example['output'], cmap=cmap, norm=norm)  # Output in the second column
            axes[row, 0].axis('off')  # Hide the axis for the input
            axes[row, 1].axis('off')  # Hide the axis for the output
    else:
        axes[0].imshow(examples[0]['input'], cmap=cmap, norm=norm)   # Input in the first column
        axes[1].imshow(examples[0]['output'], cmap=cmap, norm=norm)  # Output in the second column
        axes[0].axis('off')  # Hide the axis for the input
        axes[1].axis('off')  # Hide the axis for the output


    plt.tight_layout()
    plt.show()

def plot_test(task):
    plot_task(task, train_or_test="test")

def create_grids(grid_size, num_rectangles, primitive_functions):
    """
    Creates input and output grids with rectangles and applies primitive functions.

    Parameters:
    - grid_size: tuple (height, width) of the grid.
    - num_rectangles: number of rectangles to generate.
    - primitive_functions: list of primitive transformations to apply.

    Returns:
    - input_grid: numpy array representing the input grid.
    - output_grid: numpy array representing the output grid.
    """

    input_grid = np.zeros(grid_size, dtype=int)
    output_grid = np.zeros(grid_size, dtype=int)

    height, width = grid_size

    for _ in range(num_rectangles):
        # Random rectangle size
        rect_w = random.randint(1, width // 2)
        rect_h = random.randint(1, height // 2)

        # Random position
        x0 = random.randint(0, width - rect_w)
        y0 = random.randint(0, height - rect_h)

        # Fill the rectangle with 1s in the input grid
        input_grid[y0:y0+rect_h, x0:x0+rect_w] = 1

        # Extract the rectangle
        rect = np.zeros_like(input_grid)
        rect[y0:y0+rect_h, x0:x0+rect_w] = 1

        # Apply transformations
        transformed_rect = rect.copy()

        for func in primitive_functions:
            if func == 'translate':
                # Random translation
                dx = random.randint(-width // 4, width // 4)
                dy = random.randint(-height // 4, height // 4)
                transformed_rect = translate(transformed_rect, dx, dy)
            elif func == 'rotate':
                # Random angle
                angle = random.choice([0, 90, 180, 270])
                transformed_rect = rotate(transformed_rect, angle)
            elif func == 'mirror':
                axis = random.choice(['horizontal', 'vertical'])
                transformed_rect = mirror(transformed_rect, axis)
            elif func == 'flip':
                axis = random.choice(['horizontal', 'vertical'])
                transformed_rect = flip(transformed_rect, axis)
            elif func == 'scale':
                sx = random.uniform(0.5, 1.5)
                sy = random.uniform(0.5, 1.5)
                transformed_rect = scale(transformed_rect, sx, sy)
            else:
                raise ValueError(f"Unknown primitive function: {func}")

        # Combine the transformed rectangle with the output grid
        output_grid = np.logical_or(output_grid, transformed_rect).astype(int)

    return input_grid, output_grid

def translate(image, dx, dy):
    matrix = np.array([[1, 0, -dx],
                       [0, 1, -dy],
                       [0, 0, 1]])
    return affine_transform(image, matrix, order=0, mode='constant', cval=0)

def rotate(image, angle):
    if angle == 0:
        return image
    else:
        return np.rot90(image, k=angle // 90)

def mirror(image, axis):
    if axis == 'horizontal':
        return np.flipud(image)
    elif axis == 'vertical':
        return np.fliplr(image)
    else:
        raise ValueError("Axis must be 'horizontal' or 'vertical'")

def flip(image, axis):
    return mirror(image, axis)

def scale(image, sx, sy):
    matrix = np.array([[1/sx, 0, 0],
                       [0, 1/sy, 0],
                       [0,   0, 1]])
    output_shape = (
        int(image.shape[0] * sy),
        int(image.shape[1] * sx)
    )
    scaled_image = affine_transform(
        image,
        matrix,
        output_shape=output_shape,
        order=0,
        mode='constant',
        cval=0
    )
    # Crop or pad to match original size
    scaled_image = resize_to_grid(scaled_image, image.shape)
    return scaled_image

def resize_to_grid(image, grid_shape):
    """Resize image to match grid shape by cropping or padding."""
    result = np.zeros(grid_shape, dtype=int)
    min_rows = min(image.shape[0], grid_shape[0])
    min_cols = min(image.shape[1], grid_shape[1])
    result[:min_rows, :min_cols] = image[:min_rows, :min_cols]
    return result

def create_pattern(size=(3,3)):
    width, height = size
    pattern = np.zeros(size, dtype=int)
    nb_min =4
    while np.sum(pattern) < nb_min:
        for i in range(0, height):
            for j in range(0, width):
                value = random.randint(0, 1)
                pattern[i,j]=value

    return pattern

def translate_pattern(pattern, pattern_center,dx, dy):
    new_pattern_center = pattern_center[0] + dx, pattern_center[1] + dy
    return new_pattern_center,pattern

def rotate_pattern(pattern, pattern_center, angle):
    if angle == 0:
        return pattern_center,pattern
    else:
        return pattern_center,np.rot90(pattern, k=angle // 90)

def mirror_pattern(pattern, pattern_center, axis):
    if axis == 'horizontal':
        return pattern_center,np.flipud(pattern)
    elif axis == 'vertical':
        return pattern_center,np.fliplr(pattern)
    else:
        raise ValueError("Axis must be 'horizontal' or 'vertical'")

grid_size = (11, 11)
num_rectangles = 1
""" primitive_functions = ['translate', 'rotate', 'mirror', 'scale']
primitive_functions = ['translate','rotate']
input_grid, output_grid = create_grids(grid_size, num_rectangles, primitive_functions)

fun_test ={
    "train": [
        {
            "input": input_grid.tolist(),
            "output": output_grid.tolist()
        }
    ]
}
 """


grid_size=(11,11)
pattern_offset = 1

nb_tests = 1000
nb_examples = 3

for i in tqdm(range(0,nb_tests)):
    trains=[]
    dx = random.randint(-grid_size[0] // 4, grid_size[0] // 4)
    dy = random.randint(-grid_size[0] // 4, grid_size[0] // 4)
    dangle = random.choice([90, 180, 270])
    lambda_translate = lambda pattern: translate_pattern(pattern, pattern_center,dx, dy)
    lambda_rotate = lambda pattern: rotate_pattern(pattern,pattern_center, dangle)
    primitive_functions=[lambda_translate,lambda_rotate]
    num_primitives = random.randint(1, len(primitive_functions))
    selected_primitives = random.sample(primitive_functions, num_primitives)
    for j in range(0, nb_examples+1):
        pattern = create_pattern()
        pattern_center_dx = random.randint(-grid_size[0] // 4, grid_size[0] // 4)+grid_size[0]//2
        pattern_center_dy = random.randint(-grid_size[0] // 4, grid_size[0] // 4)+grid_size[1]//2
        pattern_center_dx=6
        pattern_center_dy=6
        pattern_center = (pattern_center_dx, pattern_center_dy)

        input_grid = np.zeros(grid_size, dtype=int)
        output_grid = np.zeros(grid_size, dtype=int)
        input_grid[
                pattern_center[0]-pattern_offset:pattern_center[0]+pattern_offset+1, 
                pattern_center[1]-1:pattern_center[1]+pattern_offset+1
                ] = pattern
        

        for primitive in selected_primitives:
            pattern_center, pattern = primitive(pattern)

        output_grid[
            pattern_center[0]-pattern_offset:pattern_center[0]+pattern_offset+1, 
            pattern_center[1]-1:pattern_center[1]+pattern_offset+1] = pattern
        
        trains.append({
            "input": input_grid.tolist(),
            "output": output_grid.tolist()
        })
    fun_test ={
        "train": trains[:nb_examples],
        "test": [trains[nb_examples-1]]
    }
    file_path = os.path.join(fun_data_dir, f'train_{i}.json')
    with open(file_path, 'w') as f:
        json.dump(fun_test, f)
    #plot_task(fun_test)

test_path = os.path.join(fun_data_dir, f'train_{1}.json')
test = json.load(open(test_path))

# test_path =r'C:\Users\mmerl\projects\ARC-AGI\data\training\0a938d79.json'

# with open(test_path) as fp:
#     train_challenges = json.load(fp)

plot_task(test)
plot_task(test,"test")

#plot_task(train_challenges,"test")
