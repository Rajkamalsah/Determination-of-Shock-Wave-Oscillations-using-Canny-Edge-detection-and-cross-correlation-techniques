# Image Edge Detection and Visualization

This project involves preprocessing images, detecting edges, extracting edge locations, saving these locations to a file, and visualizing the results. The project uses OpenCV for image processing, NumPy for numerical operations, and Matplotlib for visualization.

## Features

- **Image Preprocessing**: Load and resize images.
- **Edge Detection**: Detect edges in images using Gaussian Blur and Canny edge detection.
- **Edge Location Extraction**: Extract the coordinates of edge pixels.
- **Save Edge Locations**: Save the edge locations to a CSV file.
- **Visualization**: Plot the original image, detected edges, and re-plot saved edge locations.

## Tools and Technologies

- **Python**
- **OpenCV**
- **NumPy**
- **Matplotlib**
- **OS**

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/image-edge-detection.git
    cd image-edge-detection
    ```

2. Install the required libraries:
    ```bash
    pip install numpy opencv-python matplotlib
    ```

## Usage

1. **Preprocess Image**:
    ```python
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    def preprocess_image(image_path):
        # Load the image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Check if the image was loaded successfully
        if image is None:
            print(f"Error: Unable to load image at {image_path}")
            return None
        
        # Resize the image
        resized_image = cv2.resize(image, (640, 480))
        
        return resized_image
    ```

2. **Edge Detection**:
    ```python
    def detect_edges(image):
        # Apply Gaussian Blur to reduce noise
        blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred_image, 100, 200)
        
        return edges
    ```

3. **Extract Edge Locations**:
    ```python
    def extract_edge_locations(edges):
        # Find the coordinates of the edge pixels
        edge_locations = np.column_stack(np.where(edges > 0))
        
        return edge_locations
    ```

4. **Save Edge Locations**:
    ```python
    def save_edge_locations(edge_locations, output_path):
        # Save the edge locations to a file
        np.savetxt(output_path, edge_locations, fmt='%d', delimiter=',', header='y,x', comments='')
    ```

5. **Plot Edges**:
    ```python
    def plot_edges(image, edges):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title('Original Image')
        plt.imshow(image, cmap='gray')
        
        plt.subplot(1, 2, 2)
        plt.title('Detected Edges')
        plt.imshow(edges, cmap='gray')
        
        plt.show()
    ```

6. **Replot Edge Locations**:
    ```python
    def replot_edge_locations(csv_path):
        # Load the edge locations from the CSV file
        edge_locations = np.loadtxt(csv_path, delimiter=',', skiprows=1)
        
        # Create a blank image to plot the edges
        blank_image = np.zeros((480, 640), dtype=np.uint8)
        
        # Plot the edge locations on the blank image
        for loc in edge_locations:
            y, x = int(loc), int(loc)
            blank_image[y, x] = 255
        
        # Display the re-plotted edges
        plt.figure(figsize=(5, 5))
        plt.title('Re-plotted Edges')
        plt.imshow(blank_image, cmap='gray')
        plt.show()
    ```

7. **Example Usage**:
    ```python
    # Example usage
    directory = 'E:/github/Project_2'
    filename = 'test_image.png'
    image_path = os.path.join(directory, filename)
    output_path = 'E:/github/Project_2/edge_locations.csv'
    image = preprocess_image(image_path)

    if image is not None:
        edges = detect_edges(image)
        edge_locations = extract_edge_locations(edges)
        save_edge_locations(edge_locations, output_path)
        plot_edges(image, edges)
        print(f"Edge locations saved to {output_path}")

    # Replot the saved edge locations
    replot_edge_locations(output_path)
    ```
