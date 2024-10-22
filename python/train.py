'''
we defined measures
forms:
- connected pattern
- loose connected patterns
-know forms:
-lines
-squares
- multiple colors

About form
- closed form
- has inside

nb_pixels

colors
- nb_colors
- color_histogramm
- 

compare all inputs


output<->input
- found pattern
-   >1 ?
check correlations input<->output

'''

import numpy as np
from scipy.ndimage import label,binary_fill_holes

def find_connected_patterns(grid, connectivity='diagonal'):
    """
    Finds connected patterns in a grid based on specified connectivity.

    Parameters:
    - grid: 2D numpy array of 0s and 1s.
    - connectivity: 'diagonal', 'cardinal', or 'full'.

    Returns:
    - labeled_grid: 2D numpy array with unique labels for connected components.
    - num_features: Number of connected components found.
    """
    if connectivity == 'diagonal':
        structure = np.array([[1, 0, 1],
                              [0, 1, 0],
                              [1, 0, 1]])
    elif connectivity == 'cardinal':
        structure = np.array([[0, 1, 0],
                              [1, 1, 1],
                              [0, 1, 0]])
    elif connectivity == 'full':
        structure = np.array([[1, 1, 1],
                              [1, 1, 1],
                              [1, 1, 1]])
    else:
        raise ValueError("Connectivity must be 'diagonal', 'cardinal', or 'full'.")

    labeled_grid, num_features = label(grid, structure=structure)
    return labeled_grid, num_features

def find_lines_in_labeled_grid(labeled_grid):
    """
    Identifies which connected components in a labeled grid are lines.

    Parameters:
    - labeled_grid: 2D numpy array with labeled connected components.

    Returns:
    - lines_labels: A list of labels that correspond to lines.
    - lines_info: A dictionary with label as key and line type ('horizontal', 'vertical', 'diagonal', 'anti-diagonal') as value.
    """
    lines_labels = []
    lines_info = {}
    labels = np.unique(labeled_grid)
    labels = labels[labels != 0]  # Exclude background label 0

    for label_value in labels:
        # Get coordinates of the current connected component
        y_coords, x_coords = np.where(labeled_grid == label_value)
        points = np.column_stack((x_coords, y_coords))  # (x, y) pairs

        # Skip single-point components
        if len(points) < 2:
            continue

        # Check for horizontal line (all y the same)
        if np.all(y_coords == y_coords[0]):
            lines_labels.append(label_value)
            lines_info[label_value] = 'horizontal'
            continue

        # Check for vertical line (all x the same)
        if np.all(x_coords == x_coords[0]):
            lines_labels.append(label_value)
            lines_info[label_value] = 'vertical'
            continue

        # Check for diagonal line (x - y = c)
        if np.all((x_coords - y_coords) == (x_coords[0] - y_coords[0])):
            lines_labels.append(label_value)
            lines_info[label_value] = 'diagonal'
            continue

        # Check for anti-diagonal line (x + y = c)
        if np.all((x_coords + y_coords) == (x_coords[0] + y_coords[0])):
            lines_labels.append(label_value)
            lines_info[label_value] = 'anti-diagonal'
            continue

        # Optional: Check for other straight lines using linear regression
        # You can implement this if needed for lines at arbitrary angles

    return lines_labels, lines_info

def count_neighbors(component):
    """
    Counts the number of neighbors for each pixel in the component.

    Parameters:
    - component: 2D binary numpy array of a connected component.

    Returns:
    - neighbor_count: 2D numpy array with the neighbor counts.
    """
    # Define 4-connectivity structure
    structure = np.array([[0, 1, 0],
                          [1, 1, 1],
                          [0, 1, 0]])

    # Use convolution to count neighbors
    from scipy.ndimage import convolve
    neighbor_count = convolve(component, structure, mode='constant', cval=0)
    neighbor_count *= component  # Zero out background pixels

    return neighbor_count

def find_closed_forms_in_labeled_grid(labeled_grid):
    """
    Identifies which connected components in a labeled grid are closed forms.

    Parameters:
    - labeled_grid: 2D numpy array with labeled connected components.

    Returns:
    - closed_forms_labels: A list of labels that correspond to closed forms.
    - closed_forms_info: A dictionary with label as key and form information as value.
    """
    closed_forms_labels = []
    closed_forms_info = {}
    labels = np.unique(labeled_grid)
    labels = labels[labels != 0]  # Exclude background label 0

    for label_value in labels:
        # Extract the binary image of the current connected component
        component = (labeled_grid == label_value).astype(int)
        # Get the coordinates of the component pixels
        y_coords, x_coords = np.where(component)
        points = np.column_stack((x_coords, y_coords))  # (x, y) pairs

        # Skip components with too few pixels to form a closed shape
        if len(points) < 3:
            continue

        # Check for thin loops where all pixels have degree 2
        degrees = []
        for y, x in zip(y_coords, x_coords):
            # Count the number of neighbors in 8-connectivity
            neighbors = component[max(y - 1, 0): y + 2, max(x - 1, 0): x + 2]
            degree = np.sum(neighbors) - 1  # Subtract 1 to exclude the pixel itself
            degrees.append(degree)

        degrees = np.array(degrees)

        if np.all(degrees >= 2):
            # All nodes have degree 2, so it's a closed loop
            closed_forms_labels.append(label_value)
            closed_forms_info[label_value] = 'closed loop (thin)'
            continue

        # For solid components, check if the perimeter is significantly smaller than the area
        area = np.sum(component)
        perimeter = np.sum(component * (4 - count_neighbors(component)))
        compactness = (perimeter ** 2) / area if area > 0 else np.inf

        # Heuristic threshold for compactness (adjust as needed)
        if compactness < 20:
            closed_forms_labels.append(label_value)
            closed_forms_info[label_value] = 'closed shape (solid)'
        else:
            # Not a closed form
            continue

    return closed_forms_labels, closed_forms_info

def count_zeros_inside_closed_forms(labeled_grid, closed_forms_labels):
    """
    Counts the number of zeros (background pixels) inside each closed form in the labeled grid.

    Parameters:
    - labeled_grid: 2D numpy array with labeled connected components.
    - closed_forms_labels: List of labels that correspond to closed forms.

    Returns:
    - zeros_inside: A dictionary with label as key and number of zeros inside the closed form as value.
    """
    zeros_inside = {}

    for label_value in closed_forms_labels:
        # Extract the binary image of the current closed form
        component = (labeled_grid == label_value).astype(int)

        # Fill holes inside the component
        filled_component = binary_fill_holes(component)

        # Find the holes by subtracting the original component from the filled component
        holes = filled_component.astype(int) - component

        # Count the number of ones in 'holes', which correspond to zeros inside the closed form
        num_zeros_inside = np.sum(holes)

        zeros_inside[label_value] = num_zeros_inside

    return zeros_inside


def split_by_colors(grid):
    """
    Splits a grid into multiple grids based on unique colors.

    Parameters:
    - grid: XX numpy array with colors represented by integers.

    Returns:
    - grids: A dictionary with color as key and corresponding grid as value.
    """
    grids = {}
    unique_colors = np.unique(grid)

    for color in unique_colors:
        if color == 0:
            continue  # Skip background color
        grids[color] = (grid == color).astype(int)

    return grids

# Example usage
if __name__ == "__main__":
    grid = np.array([
        [0, 1, 1, 1, 0],
        [1, 1, 0, 0, 1],
        [0, 0, 1, 1, 0],
        [1, 0, 1, 0, 0]
    ])

    labeled_grid, num_features = find_connected_patterns(grid,'full')

    print("Number of connected patterns:", num_features)
    print("Labeled grid:")
    print(labeled_grid)

    lines_labels, lines_info = find_lines_in_labeled_grid(labeled_grid)

    print("Labeled grid:")
    print(labeled_grid)
    print("Lines labels:", lines_labels)
    print("Lines info:", lines_info)

    closed_forms_labels, closed_forms_info = find_closed_forms_in_labeled_grid(labeled_grid)
    print("Closed forms labels:", closed_forms_labels)
    print("Closed forms info:", closed_forms_info)

