import configparser
import cv2
import numpy as np

config = configparser.ConfigParser()
config.read("src\config.ini")
config = config["SAMPLE THRESHOLD"]

def sample_threshold():
    """Performs a sample mask threshold for one layer. Uses settings from the
    config.ini file.
    """    
    raw_img_path = config["path"]
    show_raw = config["show_raw"] == "True" or config["show_raw"] == "TRUE"
    raw_img = cv2.imread(raw_img_path, 0)
    # Resize the image
    width = min(int(config["max_image_width"]), raw_img.shape[1])
    factor = raw_img.shape[0] / raw_img.shape[1]
    raw_img_resized = cv2.resize(raw_img, (width, int(width * factor)))
    # Show the image
    if show_raw:
        cv2.imshow("Raw Image", raw_img_resized)
        cv2.waitKey(0)
    # Mask the image
    show_mask = bool(config["show_mask"])
    if show_mask:
        c = int(config["mask_c_value"])
        kernel = np.ones((5,5), np.uint8)
        # Blur with OTSU threshold
        blur = cv2.medianBlur(raw_img_resized, 5)
        mask = cv2.adaptiveThreshold(
            blur, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 
            151,
            c
        )
        # Open and close mask to reduce noise contours
        morph = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        cv2.imshow("Mask Image, C-Value: %i" % c, morph)
        cv2.waitKey(0)

if __name__ == "__main__":
    sample_threshold()