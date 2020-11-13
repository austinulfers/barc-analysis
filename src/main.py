from ct_layer import ct_layer
from ct_layer import target_area
from ct_material import ct_material
import json
import configparser

config = configparser.ConfigParser()
config.read("config.ini")

def run_single_layer(config: dict):
    """Runs a single layer from the scan based on the settings from the config
    file.

    Args:
        config (dict): the pertaining config settings
    """    
    config = config["SINGLE LAYER"]
    img_folder = config["IMG_FOLDER"]
    areas = target_area(img_folder)
    avg_target_area = sum(areas) / len(areas)
    layer = ct_layer(
        C=int(config["C_VALUE"]), 
        file_name=config["IMG_FILE"], 
        target_area=avg_target_area,
        export_folder=config["EXPORT_FOLDER"]
    )

def run_entire_material(config: dict):
    """Runs the entire material based on a folder of tif files based on the
    settings from the config file.

    Args:
        config (dict): the pertaining config settings
    """    
    config = config["ENTIRE MATERIAL"]
    material = ct_material(
        thresh_C=int(config["C_VALUE"]), 
        folder=config["IMG_FOLDER"], 
        mask_folder=config["EXPORT_FOLDER"]
    )
    if config["EXPORT_LAYER_JSON"].lower() == "true":
        # Save each layer's information into a JSON file
        with open("layers.json", "w") as f:
            json.dump(material.layers, f, indent=4)
    if config["EXPORT_LAYER_MATCHES"].lower() == "true":
        # Save each layer specific information with group numbers
        material.matched.to_excel("matches.xlsx")
    if config["EXPORT_LAYER_AGGREGATE"].lower() == "true":
        # Save aggregated defect information
        material.grouped.to_excel("grouped.xlsx")

if __name__ == "__main__":
    """Runs the application based on the settings from the config file.
    """    
    if config["APPLICATION"]["RUN_SINGLE_LAYER"].lower() == "true":
        run_single_layer(config)
    if config["APPLICATION"]["RUN_ENTIRE_MATERIAL"].lower() == "true":
        run_entire_material(config)