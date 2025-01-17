import os
import cv2
import pandas as pd
import numpy as np
from sklearn.cluster import MeanShift
from collections import Counter
import matplotlib.pyplot as plt


def load_images(keyword): #Type:ignore
    """
    Load and allow navigation through images of a given category from the 'data' directory.
    """
    # Get the absolute path of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct the absolute path to the 'data' directory
    data_path = os.path.join(script_dir, 'data')
    
    # Check if the 'data' directory exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Error: The directory '{data_path}' does not exist.")
    
    # Search for the specified category
    for category in os.listdir(data_path):
        if category.lower() == keyword.lower():
            category_path = os.path.join(data_path, category)
            image_files = [os.path.join(category_path, img) for img in os.listdir(category_path) if img.endswith(('.jpg', '.png', '.jpeg'))]
            if not image_files:
                print(f"No images found for category: {keyword}")
                return
            
            # Display the images in a loop with navigation controls
            print(f"Displaying images for category: {keyword}")
            index = 0
            while True:
                # Load and display the current image
                img = cv2.imread(image_files[index])
                if img is not None:
                    img_resized = cv2.resize(img, (400, 400))
                    cv2.imshow(f"Category: {category} (Use ←/→ to navigate, ESC to exit)", img_resized)
                
                # Wait for a key press
                key = cv2.waitKey(0) & 0xFF  # Mask for 8-bit keycode
                if key == 27:  # ESC key to exit
                    cv2.destroyAllWindows()
                    return
                elif key == ord('a') or key == 81:  # Left arrow key or 'a'
                    index = (index - 1) % len(image_files)  # Go to the previous image
                elif key == ord('d') or key == 83:  # Right arrow key or 'd'
                    index = (index + 1) % len(image_files)  # Go to the next image
            
            cv2.destroyAllWindows()
            return
    
    print(f"No category found for: {keyword}")

def find_largest_cluster(label_counts: Counter) -> int:
    """
    Find the label of the largest cluster.
    """
    return max(label_counts, key=label_counts.get)


def recommend_items(x, y, labels, cluster_centers, name_list) -> None:
    """
    Generate and display recommendations based on clustering results.
    """
    label_counts = Counter(labels)
    max_label = find_largest_cluster(label_counts)

    # Filter recommendations based on the largest cluster
    suggest_indices = [i for i, label in enumerate(labels) if label == max_label]

    suggested_items = [y[i][0] for i in suggest_indices]
    suggested_names = [name_list[i] for i in suggest_indices]

    # Display recommendations
    print("\nRecommendations:")
    for item_id, item_name in zip(suggested_items[:8], suggested_names[:]):
        print(f"Item ID: {item_id}   Item Name: {item_name}")

    # Visualize clusters
    colors = ['r.', 'g.', 'b.', 'c.', 'k.', 'y.', 'm.']
    while len(colors) < len(set(labels)):
        colors.extend(colors)  # Extend colors if clusters exceed color options

    for i in range(len(x)):
        plt.plot(x[i][0], x[i][1], colors[labels[i] % len(colors)], markersize=10)

    # Highlight cluster centers
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker="x", s=150, c="black", linewidths=3, zorder=10)
    plt.title("Cluster Visualization")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show ()


def main():
    """
    Main function to handle user input, image loading, and product recommendation.
    """
    keyword = input("Search for a product category: ")
    load_images(keyword)

    # Load the dataset
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_path = os.path.join(script_dir, 'person.csv')
        dataset = pd.read_csv(dataset_path)
    except FileNotFoundError:
        print("Error: 'person.csv' file not found.")
        return
    except pd.errors.EmptyDataError:
        print("Error: 'person.csv' file is empty.")
        return

    # Extract features and item details
    X = dataset.iloc[:, [2, 3]].values  # Features
    y = dataset.iloc[:, [0]].values  # Item IDs
    name_list = dataset['Item_names'].tolist()

    # Perform clustering
    ms = MeanShift()
    ms.fit(X)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_

    # Generate recommendations
    recommend_items(X, y, labels, cluster_centers, name_list)


if __name__ == "__main__":
    main()

