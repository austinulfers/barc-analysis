import plotly.graph_objects as go
import numpy as np
import pandas as pd
from collections import deque

class DefectLayer:

    def __init__(self, x: list, y:list, z: int = 0):
        """Creates the bottom layer of the defect.

        Args:
            x (list): List of x coordinates that make up the base polygon.
            y (list): List of y coordinates that make uo the base polygon.
            z (int): The z value that corresponds to this base level. Defaults to 0.
        """       
        assert len(x) == len(y), "the length of the arrays are not equal" 
        self.x = x
        self.y = y
        self.z = len(self.x) * [z]
        self.poly_cnt = len(self.x)

    def get_mesh(self):
        # Getting meshes for the sides of the layer
        x = 2 * self.x
        y = 2 * self.y
        z = self.z.copy() + [i + 1 for i in self.z]
        i = list(np.arange(len(x)))
        front_j = deque(np.arange(len(x)/2))
        front_j.rotate(-1)
        front_k = deque(np.arange(len(x)/2, len(x)))
        back_j = front_k.copy()
        back_j.rotate(-1)
        front_k = back_j.copy()
        front_k.rotate()
        j = [int(l) for l in front_j + back_j]
        k = [int(l) for l in front_k + front_j]
        # Getting the meshes for the top and bottom of the layer
        x_mean = sum(x)/len(x)
        x.append(x_mean)
        x.append(x[-1])
        y_mean = sum(y)/len(y)
        y.append(y_mean)
        y.append(y[-1])
        z.append(z[0])
        z.append(z[-2])
        i += i
        j += j
        k += int((len(x)-2)/2) * [x.index(x_mean)]
        k += int((len(x)-2)/2) * [y.index(y_mean) + 1]
        return x, y, z, i, j , k

def render(x, y, z, i, j , k):
    """Renders the current layer within plotly.
    """
    fig = go.Figure(
        data=[
            go.Mesh3d(
                x = x,
                y = y,
                z = z,
                i = i,
                j = j,
                k = k
            )  
        ],
        layout=go.Layout(
            scene=dict(
                aspectmode="data",
                aspectratio=dict(x=1, y=1, z=1)
            )
        )
    )

    fig.show()

def merge(d1: DefectLayer, x2, y2, z2, i2, j2, k2):
    x1, y1, z1, i1, j1, k1 = d1.get_mesh()
    x = x1 + x2
    y = y1 + y2
    z = z1 + z2
    new_i2 = [l + len(x1) for l in i2]
    i = i1 + new_i2
    new_j2 = [l + len(x1) for l in j2]
    j = j1 + new_j2
    new_k2 = [l + len(x1) for l in k2]
    k = k1 + new_k2
    return x, y, z, i, j, k

def construct(defect_n: int, mat=None, mat_matches: pd.DataFrame = None, mat_layers: list = None):
    """Takes in a CT scanned material and returns the polygons needed to reconstruct a specific
    defect within the material. Works in a way that allows for it to be either ran entirely or from
    a file. Either mat must be non-None or both mat_matches and mat_layers must be non-None but not
    both.

    Args:
        defect_n (int): A grouped defect number.
        mat (Material): The material that is being evaluated. Defaults to None.
        mat_matches (pd.Dataframe): A material's match's dataframe. Defaults to None.
        mat_layers (list): A material's list of layers. Defaults to None.
    """
    # Checking that conflicting arguments weren't passed
    operate = bool(mat)
    load = mat_matches is not None and bool(mat_layers)
    if operate and load:
        raise ValueError("Both mat and (mat_matches and mat_layers) were passed. Please choose one.")
    if mat:
        defect_matches = mat.matched[mat.matched["defect_n"] == defect_n]
        defect_layers = mat.layers
    elif load:
        defect_matches = mat_matches
        defect_layers = mat_layers
    else:
        raise ValueError("A material or a loaded version was not passed.")
    # Ensuring the passed defect number is a match within the data set
    if not len(defect_matches) > 0:
        raise BaseException("Defect Number: %i is not a valid defect number." % defect_n)
    search_layers = list(defect_matches[defect_matches["defect_n"] == defect_n]["z"])
    defect_matches = defect_matches[defect_matches["defect_n"] == defect_n]
    found_polygons = []
    for search in search_layers:
        layer = defect_layers[search]
        assert layer["z value"] == search, "layer z: %i is not search: %i" % (layer["z value"], search)
        match = defect_matches[defect_matches["z"] == search]
        assert len(match) == 1, "match was found to have more than 1 row."
        for defect in layer["defects"]:
            # TODO: This check doesn't work.
            dist = float(np.sqrt(((defect["x"]-match["x"])**2)+((defect["y"]-match["y"])**2)))
            if dist < 2:
                found_polygons.append({"z": layer["z value"], "polygon": defect["polygon"]})
    for defect, n in zip(found_polygons, range(len(found_polygons))):
        defect_layer = DefectLayer(
            x = [d[0][0] for d in defect["polygon"]],
            y = [d[0][1] for d in defect["polygon"]],
            z = defect["z"]
        )
        if n == 0:
            x, y, z, i, j, k = defect_layer.get_mesh()
        else:
            x, y, z, i, j, k = merge(defect_layer, x, y, z, i, j, k)
    return x, y, z, i, j, k
                    
if __name__ == "__main__":
    import json
    curr = "1-2"
    defect = 1976
    matches = pd.read_excel("C:/Users/austi/OneDrive/UW/BARC/ct-scan/exported data/reg_CT/%s_matches.xlsx" % curr, index_col=0)
    with open("C:/Users/austi/OneDrive/UW/BARC/ct-scan/exported data/reg_CT/%s_layers.json" % curr) as infile:
        layers = json.load(infile)
    x, y, z, i, j, k =construct(
        defect_n = defect,
        mat_matches = matches,
        mat_layers = layers
    )
    render(x, y, z, i, j, k)
    