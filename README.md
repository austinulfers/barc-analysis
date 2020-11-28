# barc-analysis [![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/austinulfers/barc-analysis/main/src/inspect_material.py)

Material Analysis & Model Creation for BARC

## Installing Dependencies

Please follow the steps below to in order setup and use the resources within the repo correctly.

1. Ensure you are using **Python 3.8.6**. To check which version you are using, use  the code `python -V` within the console.
2. Although not required, it is best practice to always use a virtual
environment. Use `python -m venv env` within the directory to get a copy of your
existing environment.
3. Install all the required packages using the command
`pip install -r requirements.txt`.
4. Open your bash console and activate the environment using the following command `./env/scripts/activate`

## Material Thresholding

1. Figuring out the correct thresholding value (C Value) is crucial to the success for this application to work successfully. I created a simple tool to make this easier. After you have installed the above dependencies within your `env` folder and activated it, use the following command to launch the thresholding application `streamlit run .\src\inspect_material.py`.
2. When it launches, you will see an error until you put a file path in the "Image Filepath" input box.
3. Here, you will see the raw image on the left and the example masked image on the right. You will want to adjust the C Value until the masked image meets the following criteria:
    - The masked image closely resembles the raw image so that you can see the major target areas and the defects contained within the specific layer.
    - If there are a lot of small defects within the raw image, increase the C Value so that the most amount of defects you see within the masked image is at max roughly 15-20 per target area. The current application implementation has a recursive limitation that will likely cause an error when grouping the defects if you have more defects than this when you later go try and run the application.
    - On the edges of the target area, you want to try and make it so the masked image can differentiate between the the background and the target area. Below is an example of a good vs bad example of this in practice.
        - Good (C Value of 3 on 5_1-1436.tif) ![Good Material Thresholding](/img/good_material_thresholding.jpg)
        - Bad (C Value of 5 on 5_1-1436.tif) ![Bad Material Thresholding](/img/bad_material_thresholding.jpg)
    - When possible, you want to balance the above criteria along with minimizing the amount of exterior noise outside of the target areas. This section is definitely more art than science so feel free to experiment with what does and doesn't work.

## Running the Application

1. Once you are satisfied with your C Value that you got the from the previous section, it is time to open the `config.ini` file and get ready to run the application.

## Defect Reconstruction

## Dealing with Errors
