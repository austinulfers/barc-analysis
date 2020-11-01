import streamlit as st
from PIL import Image
import cv2
import numpy as np

class App:

    def __init__(self):
        self.pref()
        st.title("Material Sample Thresholding")

    def pref(self):
        """Sets the page preferences for the application.
        """
        st.beta_set_page_config(layout="wide")

    def run(self):
        """Runs the application function.
        """
        col_1, col_2 = st.beta_columns(2)
        file_path = col_1.text_input("Image Filepath")
        file_path = file_path.replace("\\", "/")
        c_value = col_2.number_input("C Value", value=0, step=1)
        col_1.header("Raw Image")
        col_2.header("Masked Image")
        raw_img = cv2.imread(file_path, 0)
        col_1.image(raw_img, use_column_width=True)
        mask_img = self._mask_image(raw_img, c_value)
        col_2.image(mask_img, use_column_width=True)

    def _mask_image(self, img: np.ndarray, c: int) -> np.ndarray:
        """Returns a localized thresholding mask of the passed image.

        Args:
            img (np.ndarray): the original image
            c (int): the value for thresholding

        Returns:
            np.ndarray: the masked image
        """
        kernel = np.ones((5,5), np.uint8)
        # Blur with OTSU threshold
        blur = cv2.medianBlur(img, 5)
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
        return morph

if __name__ == "__main__":
    app = App()
    app.run()