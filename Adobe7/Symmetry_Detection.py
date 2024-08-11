import cv2
import numpy as np

# Define function to load and check image
def load_image(path):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Error: Unable to read the image file {path}")
    return image

# Define function to apply flips and show images
def apply_flips_and_display(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    flipped_horizontally = cv2.flip(gray_image, 1)
    flipped_vertically = cv2.flip(gray_image, 0)

    cv2.imshow("Original Image", image)
    cv2.imshow('Flipped Horizontally', flipped_horizontally)
    cv2.imshow('Flipped Vertically', flipped_vertically)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Define function to visualize vertical symmetry
def visualize_vertical_symmetry(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_with_symmetry = image.copy()
    cv2.line(image_with_symmetry, (gray_image.shape[1] // 2, 0), 
             (gray_image.shape[1] // 2, gray_image.shape[0]), (0, 0, 255), 2)

    cv2.imshow('Original Image', image)
    cv2.imshow('Vertical Symmetry', image_with_symmetry)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Define function to visualize horizontal symmetry
def visualize_horizontal_symmetry(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_with_symmetry = image.copy()
    cv2.line(image_with_symmetry, (0, gray_image.shape[0] // 2),
             (gray_image.shape[1], gray_image.shape[0] // 2), (255, 0, 255), 2)

    cv2.imshow('Original Image', image)
    cv2.imshow('Horizontal Symmetry', image_with_symmetry)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Define function to visualize diagonal symmetry
def visualize_diagonal_symmetry(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_with_symmetry = image.copy()
    height, width = gray_image.shape
    cv2.line(image_with_symmetry, (0, 0), (width, height), (0, 255, 0), 2)  # Diagonal from top-left to bottom-right
    cv2.line(image_with_symmetry, (0, height), (width, 0), (0, 255, 0), 2)  # Diagonal from bottom-left to top-right

    cv2.imshow('Original Image', image)
    cv2.imshow('Symmetry Visualization', image_with_symmetry)
    key = cv2.waitKey(0)
    if key == 27:  # Escape key
        cv2.destroyAllWindows()
    elif key == ord('s'):  # 's' key
        cv2.imwrite('E:/img1_symmetry.jpg', image_with_symmetry)
        cv2.destroyAllWindows()
    print('Operation completed successfully')

# Main execution
if __name__ == "__main__":
    # Example usage of the functions
    img_path = 'E:/img1.jpg'
    img = load_image(img_path)

    # Apply flips and display results
    apply_flips_and_display(img)

    # Visualize vertical symmetry
    visualize_vertical_symmetry(img)

    # Visualize horizontal symmetry
    visualize_horizontal_symmetry(img)

    # Visualize diagonal symmetry
    visualize_diagonal_symmetry(img)
