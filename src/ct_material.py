from ct_layer import ct_layer
from ct_layer import target_area
import os
from tqdm import tqdm
import pandas as pd
import json
import numpy as np
import cv2
import re
from barc.ctscan.Reconstruct import construct, render

MAX_DIST_THRESH = 15
MAX_LAST_LAYER_SURFACE_AREA = 100

class ct_material(object):
    """The material class that takes in z layers of ct-scans and performs operations on them. Works
    in part with the ZLayer class.
    """    
    def __init__(self, thresh_C, folder, mask_folder=None, area_thresh=None, dist_thresh=MAX_DIST_THRESH, min_zval = 0):
        """Initalizes the class by passing in the folder where the images are contained as well as
        attributes of this specific material.

        Args:
            thresh_C (int): the thresholding constant
            folder (str): filpath of the folder where the images are stored.
            mask_folder (str, optional): filepath for where to save the masked images, if None, it
            will not save the masked images. Defaults to None.
            area_thresh (dict, optional): dict for the min and max area. Defaults to None.
            dist_thresh (int, optional): the minimum distance needed between defects to count as a
            grouping. Defaults to 5.

        Raises:
            AssertionError: all files in folder must contain .tif
        """
        self.dist_thresh = dist_thresh
        self.area_thresh = area_thresh
        self.min_zval = min_zval
        self.layers = []
        mask_folder = mask_folder.replace("\\", "/")
        folder = folder.replace("\\", "/")
        self.img_folder = folder
        img_fns = os.listdir(self.img_folder)
        img_fns.sort()
        for fn in img_fns:
            if ".tif" not in fn:
                print(fn)
                raise AssertionError("not all files in directory contain .tif")
        if os.path.exists(mask_folder + "/layers.json"):
            os.remove(mask_folder + "/layers.json")
        target_areas = target_area(self.img_folder)
        avg_target_area = sum(target_areas) / len(target_areas)
        for file in tqdm(img_fns, desc="Identifying Defects"):
            if int(re.sub("[^0-9]", "", file)) >= self.min_zval:
                self.layers.append(
                    ct_layer(
                        C=thresh_C,
                        file_name=os.path.join(self.img_folder, file), 
                        target_area=avg_target_area,
                        export_folder=mask_folder, 
                        area_thresh=area_thresh
                    ).feat
                )
        if mask_folder is not None:
            self.match_defects()
            self.aggregate_defects()

    def sample_threshold(self, n):
        """Returns an averaged threshold for a random sample of images within the folder directory.
        This is mainly used to reduce the amount of time it takes for this script to run. To be more
        accurate per layer, this should be done for every layer. THIS METHOD IS DEPRECATED AND WILL
        LIKELY FAIL.

        Args:
            n (int): the number of samples to draw.

        Returns:
            float: the averaged mask threshold.
        """        
        thresholds = []
        img_fns = os.listdir(self.img_folder)
        samples = np.random.choice(img_fns, size=n, replace=False)
        for sample in tqdm(samples, desc="Averaging Mask Threshold"):
            img = cv2.imread(self.img_folder + "/" + sample, 0)
            # Utilizes KMeans for finding a relative threshold value
            m = KMeans(2).fit(img.reshape(-1, 1))
            max_index = list(m.cluster_centers_).index(m.cluster_centers_.max())
            threshold = np.percentile(img.flatten()[m.labels_ == max_index], 1.5)
            thresholds.append(threshold)
        return sum(thresholds) / len(thresholds)

    def load_layers(self, export_folder):
        """Used to load a material that has been executed in the past. This method is depreciated.

        Args:
            export_folder (str): the folder in which the layers.json file is held.
        """        
        file_path = export_folder + "/layers.json"
        file = open(file_path, "r")
        existing = file.readlines()[0]
        content = json.loads(existing)
        file.close()
        self.layers = content

    def to_df(self):
        """Returns a dataframe consisting of every layer's data.

        Returns:
            pd.DataFrame: dataframe of the raw layer data.
        """        
        df = pd.DataFrame()
        for z in tqdm(self.layers, desc="Converting to DataFrame"):
            if len(z["defects"]) == 0:
                continue
            z_val = z["z value"]
            for d in z["defects"]:
                df = df.append(
                    pd.DataFrame(
                        {
                            "z": z_val,
                            "polygon": d["polygon"],
                            "area": d["area"],
                            "diameter": d["diameter"],
                            "x": round(d["x"]),
                            "y": round(d["y"]),
                            "distance": d["distance"],
                            "hull": d["hull"],
                            "aspect": d["aspect"],
                            "circularity": d["circularity"],
                            "roundness": d["roundness"],
                            "solidity": d["solidity"],
                            "extent": d["extent"],
                            "major": d["major"],
                            "minor": d["minor"]
                        }
                    ), ignore_index=True
                )
        return df

    def layer_matches(self, prev, upco):
        """Matches defects between z layers and returns values if one is found.

        Args:
            prev (pandas.DataFrame): the base layer that has ties to all previous layers.
            upco (pandas.DataFrame): the next layer that has defects within it.

        Returns:
            float: distance between defect if a match is located.
            bool: true if a match is found, false otherwise.
            int: the index of the defect within the upcoming layer where a defect is found.
        """        
        for i in range(len(upco)):
            if type(upco) != "pandas.core.series.Series":
                row = upco.iloc[i,:]
            else:
                row = upco
            dist = np.sqrt((prev["x"] - row["x"])**2 + (prev["y"] - row["y"])**2)
            if dist < self.dist_thresh:
                return dist, True, row.name
        return None, False, None

    def group_defect(self, df, group, base):
        """Recursively searches for defect groups using the layer matches method between layers.

        Args:
            df (pandas.DataFrame): the current subsection of the whole layers dataset
            group (pandas.DataFrame): the current grouping of defects
            base (pandas.DataFrame): the current layer that the group is searching through

        Returns:
            pandas.DataFrame: the finished grouping for a defect
        """        
        group = group.append(base)
        next_layer = df[df["z"] == base["z"] + 1]
        dist, matches, index = self.layer_matches(base, next_layer)
        if not matches:
            return group
        new_base = df.loc[index, :]
        df = df.drop(index)
        return self.group_defect(df, group, new_base)

    def match_defects(self):
        """Matches defects across z layers.
        """        
        groups = pd.DataFrame()
        df = self.to_df()
        defect_n = 0
        start_len = len(df)
        pbar = tqdm(desc="Grouping Defects", total=start_len)
        while not df.empty:
            empty_group = pd.DataFrame()
            base = df.iloc[0, :]
            df.drop(0, axis=0, inplace=True)
            group = self.group_defect(df, empty_group, base)
            group['defect_n'] = defect_n
            groups = groups.append(group)
            for index in group.index[1:]:
                df.drop(index, axis=0, inplace=True)
            df.reset_index(drop=True, inplace=True)
            defect_n += 1
            pbar.n = start_len - len(df)
            pbar.refresh()
        pbar.close()
        self.matched = groups.reset_index(drop=True)
        self.matched.drop(["polygon"], 1, inplace=True)
        self.matched.drop_duplicates(
            subset=["area", "diameter", "distance", "x", "y", "z"], 
            inplace=True
        )
        self.matched["span"] = 1

    def aggregate_defects(self):
        """Aggregates the defects based on the match defects method.
        """        
        defects = self.matched.groupby('defect_n').agg({
            "area": ["sum", "max"],
            "diameter": "max",
            "distance": "min",
            "hull": "sum",
            # TODO: make aspect, circularity, roundness, solidity, extent weighted
            "aspect": "max",
            "circularity": "mean",
            "roundness": "mean",
            "solidity": "mean",
            "extent": "mean",
            "major": "max",
            "minor": "max",
            "span": "sum",
            "x": "mean",
            "y": "mean",
            "z": "mean",
            "defect_n": "mean"
        })
        defects.columns = ['-'.join(col).strip() for col in defects.columns.values]
        defects["xz-aspect"] = defects["major-max"] / defects["span-sum"]
        defects["area-prop"] = defects["area-max"] / np.mean(self.target_area())
        defects["area-prop-rank"] = defects["area-prop"].rank(ascending=False)
        defects["dist-rank"] = defects["distance-min"].rank()
        self.grouped = defects
        self.indentify_surface_defects()

    def to_3darray(self):
        return np.array([z.mask_img for z in self.layers])

    def indentify_surface_defects(self):
        surface_defects = []
        for n in set(self.matched["defect_n"]):
            subset = self.matched[self.matched["defect_n"] == n]
            last_z = max(subset["z"])
            last_area = subset[subset["z"] == last_z]["area"].values[0]
            if last_area > MAX_LAST_LAYER_SURFACE_AREA:
                surface_defects.append(n)
        surface_column = []
        for i in self.grouped["defect_n-mean"]:
            if i in surface_defects:
                surface_column.append(1)
            else:
                surface_column.append(0)
        self.grouped["surface"] = surface_column

    def __iter__(self):
        yield from self.layers

    def __getitem__(self, n):
        return self.layers[n]

    def __len__(self):
        return len(self.layers)

    def __str__(self):
        return str(self.grouped)

    def __repr__(self):
        return str(self.grouped)

if __name__ == "__main__":
    for curr in [
        ("1-2", 9), 
        ("3-2", 9), 
        ("5-1", 3), 
        ("6-1", 7), 
        ("7-2", 9),
        ("8-2", 2), 
        ("11-1", 9), 
        ("12-1", 4), 
        ("13-2", 10), 
        ("14-2", 9), 
        ("16-1", 9), 
        ("18-1", 8),
        ("19-2", 5),
        ("20-2", 8),
        ("26-2", 9),
        ("27-2", 11)]:
        print(curr[0])
        material = ct_material(
            thresh_C=curr[1], 
            folder="E:/reg_CT/%s_Slices/raw" % curr[0], 
            mask_folder="E:/reg_CT/%s_Slices/mask" % curr[0]
        )
        # Save each layer's information into a JSON file
        with open("%s_layers.json" % curr[0], "w") as f:
            json.dump(material.layers, f, indent=4)
        # Save each layer specific information with group numbers
        material.matched.to_excel("%s_matches.xlsx" % curr[0])
        # Save aggregated defect information
        material.grouped.to_excel("%s_grouped.xlsx" % curr[0])