# BARC-Analysis

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
2. When it launches, you will see an error until you put a file path in the "Image File Path" input box.
3. Here, you will see the raw image on the left and the example masked image on the right. You will want to adjust the C Value until the masked image meets the following criteria:
    - The masked image closely resembles the raw image so that you can see the major target areas and the defects contained within the specific layer.
    - If there are a lot of small defects within the raw image, increase the C Value so that the most amount of defects you see within the masked image is at max roughly 15-20 per target area. The current application implementation has a recursive limitation that will likely cause an error when grouping the defects if you have more defects than this when you later go try and run the application.
    - On the edges of the target area, you want to try and make it so the masked image can differentiate between the the background and the target area. Below is an example of a good vs bad example of this in practice.
        - Good (C Value of 3 on 5_1-1436.tiff) ![Good Material Thresholding](/img/good_material_thresholding.jpg)
        - Bad (C Value of 5 on 5_1-1436.tiff) ![Bad Material Thresholding](/img/bad_material_thresholding.jpg)
    - When possible, you want to balance the above criteria along with minimizing the amount of exterior noise outside of the target areas. This section is definitely more art than science so feel free to experiment with what does and doesn't work.

## Running the Application

1. Once you are satisfied with your C Value that you got the from the previous section, it is time to open the `config.ini` file and get ready to run the application.

Below you will find an explanation for the variables found within the config file.

- **Application**:
  - *RUN_SINGLE_LAYER* = (TRUE/FALSE) | When you run `main.py`, setting this to true will thresholding and defect calculations for a single layer based on the settings passed in the [SINGLE LAYER] header of the config file.
  - *RUN_ENTIRE_LAYER* = (TRUE/FALSE) | When you run `main.py`, setting this to true will preform the defect thresholding and calculations for an entire series of layer images based on the settings in the [ENTIRE MATERIAL] header of the config file.
- **Single Layer** - It is important to note that these below preferences are only necessary if you set the RUN_SINGLE_LAYER to TRUE:
  - *C_VALUE* = (INTEGER) | This is the number that you will have figured out from the previous steps after trying different C values. Chances are it is a number between 1-10 however, if you skipped the Material Thresholding step above, make sure to go back and figure out the C value from that list of instructions.
  - *IMG_FOLDER* = (C:/path/to/folder/) | This will be the folder where your specific layer is stored on your computer.
  - *IMG_FILE* = (C:/path/to/image.tiff) | This is the specific image layer that you want to perform the layer calculations on.
  - *EXPORT_FOLDER* = (C:/path/to/export/folder) | This will be where the outputted layer will be exported to.
- **Entire Material** - It is important to note that these below preferences are only necessary if you set the RUN_ENTIRE_LAYER to TRUE:
  - *C_VALUE* = (INTEGER) | This is the number that you will have figured out from the previous steps after trying different C values. Chances are it is a number between 1-10 however, if you skipped the Material Thresholding step above, make sure to go back and figure out the C value from that list of instructions. This value will be applied to all layers.
  - *IMG_FOLDER* = (C:/path/to/folder) | This is the path to the folder where your series of layer images are stored. Make sure this folder only contains your image files and nothing else.
  - *EXPORT_FOLDER* = (C:/path/to/export/folder) | This is where the masked images will be saved.
  - *EXPORT_LAYER_JSON* = (TRUE/FALSE) | This file contains ALL of the information from every layer and is outputted after the entire script is complete. This file is often large because it contains all of the defect polygons as well as the target area polygons. It is exported to the project's working directory.
  - *EXPORT_LAYER_MATCHES* = (TRUE/FALSE) | This file contains layer specific data for every defect on every layer as well as which group a defect belongs to. It is exported to the project's working directory.
  - *EXPORT_LAYER_AGGREGATE* = (TRUE/FALSE) | This file contains the aggregate version of the file listed above. It merges the defects between layers so it mainly has just the defect data but not the layer specific information. It is exported to the project's working directory.
- **Hyperparameters** - The numbers currently set within the hyperparameters are found to be the best default values. Feel free to change them and see how they effect the calculations, but to be consistent make sure you keep them the same for an entire series:
  - *MAX_DEFECT_AREA* = (INTEGER (pixels), Default = 10000) | This is the max defect area of any layer specific defect. It is mainly used to restrict the program from thinking that defects that are near the surface are just large defects. It is also used to restrict the program from thinking an empty target area is a defect.
  - *MIN_DEFECT_AREA* = (INTEGER (pixels), Default = 5) | This is the opposite of the above metric. It is the minimum defect area of a layer specific defect. It is used to keep noise out of the results as well as keep the amount of layer specific defects limited when the C value isn't entirely optimized.
  - *BLOCK_SIZE* = (INTEGER, Default = 151) | This is a variable that is required when performing the thresholding, it has to be set to a value so it should serve some importance, but I have yet to find that...
  - *MAX_DIST_THRESHOLD* = (INTEGER (pixels), Default = 15) | This value is used when matching defects between layers. This is the maximum distance that the center of a defect could travel between layers. When it is too large, it matches separate defects, however, when it is too small, it splits one defect into multiple if the defect moves drastically between layers. Any XY distance between 0 and this value between layers will be considered the same defect.
  - *MAX_LAST_LAYER_SURFACE_AREA* = (INTEGER (pixels), Default = 100) | This value is the maximum area of the top most layer of a defect. It was found that when defects have large last layer area values, they tended to be considered "surface defects" that we want to exclude.

1. Now that you have your `config.ini` setup, you're ready to run the `main.py` program.

2. If you chose to run an entire material, the process can be quite time consuming. When I was running this for various materials, it wasn't uncommon for the operations to run for upwards of 4+ hours per material. This mainly has to do with the size of the layer images as well as how many defects there are within the material (i.e. materials with more defects take longer.)

## Defect Reconstruction

## Dealing with Errors
