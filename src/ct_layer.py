import numpy as np
import cv2
import re
import imutils
from imutils import contours
import pandas as pd
import json
import os
from shapely.geometry import Polygon
from shapely.geometry import Point
from shapely.geometry import MultiPolygon
from shapely.ops import cascaded_union
import math
from scipy.spatial import ConvexHull

MASK_PERCENTILE = 1.5
MAX_DEFECT_AREA = 10000
MIN_DEFECT_AREA = 5
BLOCK_SIZE = 151

def target_area(img_folder:str) -> tuple:
    """Returns the area of both of the target areas in pixels.

    Args:
        img_folder (str): path of image folder

    Returns:
        tuple: The area of both target areas.
    """
    all_files = os.listdir(img_folder)
    all_files.sort()
    assert min([len(f) for f in all_files]) > 8, "not all files in the img_folder are valid."
    file_nums = [int(f.split("_")[-1].split(".")[0]) for f in all_files]
    start_layer = min(file_nums)
    end_layer = max(file_nums)
    middle_layer = int((start_layer + end_layer) / 2)
    img = cv2.imread(img_folder + "/" + all_files[middle_layer], 0)
    blur = cv2.GaussianBlur(img, (5,5), 0)
    thresh, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cnts, hierarchy = cv2.findContours(otsu, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = np.array(cnts, dtype=object)
    moments = pd.DataFrame(hierarchy[0], columns=['next', 'prev', 'child', 'parent'])
    target_area_indexes = moments[moments["parent"] == 0].index.values
    areas = tuple([cv2.contourArea(cnts[indx]) for indx in target_area_indexes])
    return areas

class ct_layer(object):    
    """
    ct_layer(C, file_name, target_area, export_folder=None, area_thresh=None)

    Class that represents a single Z layer within a material CT-Scan.
    """

    def __init__(self, C, file_name, target_area, export_folder=None, area_thresh=None):
        """Initializes a single z layers within a materical ct-scan. 

        ### Args:
            C (int): a contant used for thresholding.
            file_name (str): path to image
            target_area (float): the average size of one of the target areas.
            export_folder (str, optional): path to export mask image. Defaults to None.
            thresh (dict, optional): dict for the min and max area. Defaults to None.
        """
        if area_thresh is not None:
            self.max_area = area_thresh["max"]
            self.min_area = area_thresh["min"]
        else:
            self.max_area = MAX_DEFECT_AREA
            self.min_area = MIN_DEFECT_AREA
        self.C = C
        self.target_area = target_area
        self.mask_img = np.array([])
        self.raw_img = np.array([])
        self.feat = {}
        self._moments: pd.DataFrame()
        self._initialize(file_name, export_folder)

    @staticmethod
    def dist(p1, p2):       
        """Returns the elucidean distance between two points.

        ### Args:
            p1 (list, tuple): point 1
            p2 (list, tuple): point 2

        ### Raises:
            BaseException: given the length of p1 or p2 is not 2

        ### Returns:
            float: distance between p1 and p2
        """    
        if len(p1) != 2 or len(p2) != 2:
            raise BaseException("Point p1 or p2 is not length 2:", p1, p2)    
        return np.sqrt((p1[0]-p2[0])**2 + (p1[1] - p2[1])**2)
    
    def _initialize(self, file_name, export_folder=None):
        """Initializes a single z layers within a materical ct-scan.

        ### Args:
            file_name (str): path to image
            export_folder (str, optional): path to export mask image. Defaults to None.
        """       
        file_name = file_name.replace("\\", "/") 
        self.feat.update({"file name": file_name})
        self.raw_img = cv2.imread(file_name, 0)
        z_val = int(re.sub("[^0-9]", "", self.feat["file name"].split("_")[-1]))
        self.feat.update({"z value": z_val})
        self._threshold(export_folder)
        self._contours()
        self._edge_dist()
        len_defects = len(self.feat["defects"])
        self._recursive_group(i=len_defects - 1, j=len_defects - 1)

    def _checkpoint(self, export_folder):
        """A way to get information from already exported masks. THIS METHOD IS
        DEPRECATED.

        Args:
            export_folder ([type]): [description]

        Raises:
            TypeError: [description]
        """        
        file_path = export_folder + "/layers.json"
        try:
            file = open(file_path, "r")
            existing = file.readlines()[0]
            content = json.loads(existing)
            to_append = self.feat
            content.append(to_append)
            file = open(file_path, "w")
            try:
                file.write(json.dumps(content))
            except TypeError:
                print(to_append)
                file.close()
                raise TypeError
            file.close()
        except FileNotFoundError:
            file = open(file_path, "w")
            file.write(json.dumps([self.feat]))
            file.close()

    def _threshold(self, export_folder:str=None) -> None:
        """Performs a threshold on the layer's raw image. Saves mask if it finds one from a previous
        run.

        Args:
            export_folder (str, optional): folder location. Defaults to None.

        Returns:
            None: returns when done
        """
        export_fn = "/" + self.feat["file name"].split("/")[-1].split(".")[0] + "_m.tif"
        # Array for 2D convolution
        kernel = np.ones((5, 5), np.uint8)
        # Blur for adaptive threshold
        blur = cv2.medianBlur(self.raw_img, 5)
        mask = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
            cv2.THRESH_BINARY_INV, BLOCK_SIZE, self.C)
        # Open and close mask to reduce noise contours
        self.mask_img = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        if export_folder is not None:
            # Export the new mask
            cv2.imwrite(export_folder + export_fn, self.mask_img)
        return None

    def _contours(self):
        """Goes through layer and identifys all contours within each layer and
        matches them to their parent contour that the defect is found within.
        """        
        # Find contours in the edge map
        cnts, hierarchy = cv2.findContours(self.mask_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = np.array(cnts, dtype=object)
        # Select only contours that are the inner most children
        self._moments = pd.DataFrame(hierarchy[0], columns=['next', 'prev', 'child', 'parent'])
        self._moments["area"] = [cv2.contourArea(c) for c in cnts]
        top_moments = self._moments[self._moments["area"] > (self.target_area * 0.75)]
        parent_indexes = top_moments[(top_moments["parent"] != -1)].index
        parent_cnts = []
        for indx, p in enumerate(cnts):
            if indx in parent_indexes:
                parent_cnts.append(p)
        child_indexes = []
        child_cnts = []
        for indx, x in enumerate(self._moments["parent"]):
            if x in parent_indexes:
                child_indexes.append(x)
                child_cnts.append(cnts[indx])
        self.feat.update({"defects": []})
        for indx, c in zip(child_indexes, child_cnts):
            self._contour(c, indx, "defects", True)
        self.feat.update({"targets": []})
        for indx, p in zip(parent_indexes, parent_cnts):
            self._contour(p, indx, "targets", False)      
    
    def _contour(self, c, indx, key, restrict):
        """Finds features of a contour.

        ### Args:
            c (numpy.ndarray): the contour polygon array
            indx (int): the matching parent child index
            key (str): keyword to deposit the contout information to
            restrict (bool): if you are working with defects.

        ### Returns:
            None: returns None given contour doesnt match criteria.
        """              
        # Area in pixels of the contour
        area = cv2.contourArea(c)
        if (area > self.max_area or area < self.min_area) and restrict:
            return None
        # Fit ellipse only works if the contour has at least 5 points
        if c.shape[0] < 5:
            return None
        # Finding the center of the contour
        box = cv2.minAreaRect(c)
        box = cv2.boxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        # Finding minimum encosing ellipse
        (x, y), (ma, mi), angle = cv2.fitEllipse(c)
        ellipse_area = math.pi * ma * mi
        h = ((ma-mi)**2)/((ma+mi)**2)
        # Ignores defect if the axis doesn't have a major or minor axis
        if ma == 0 or mi == 0:
            return None
        ellipse_peri = math.pi*(ma+mi)*(1+((3*h)/(10+math.sqrt(4-3*h))))
        # Bouding rectangle
        x, y, w, h = cv2.boundingRect(c)
        # Convex hull
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        defect = {
            "match": indx,
            "polygon": c.tolist(),
            "area": area,
            "hull": hull_area,
            "diameter": np.sqrt(area),
            "x": float(np.mean(box.T[0])),
            "y": float(np.mean(box.T[1])),
            "aspect": mi / ma,
            "circularity": 4*math.pi*(ellipse_area/(ellipse_peri**2)),
            "roundness": 4*(area/(math.pi*ma**2)),
            "solidity": float(area) / hull_area,
            "extent": float(area) / (w*h),
            "major": ma,
            "minor": mi
        }
        self.feat[key].append(defect)

    def _edge_dist(self):
        """Calcualtes each defect's distance to the edge of the target area.
        """        
        for indx, d in reversed(list(enumerate(self.feat["defects"]))):
            for t in self.feat["targets"]:
                if d["match"] == t["match"]:
                    poly = [x[0] for x in t["polygon"]]
                    target_poly = Polygon(poly)
                    defect = Point(d["x"], d["y"])
                    dist = target_poly.exterior.distance(defect) - (d["diameter"] / 2)
                    self.feat["defects"][indx].update({"distance": dist})
                    break
            # If no distance is found, the defect is removed.
            try:
                d["distance"]
            except KeyError:
                self.feat["defects"].remove(d)

    def _replace_pair(self, dist, i, j):
        """Merges two defects as one. Averages their x and y value and adds the distance to the
        defect diameter. Area of the polygons are summed.

        Args:
            dist (float): distance from defect i to defect j
            i (int): index of a defect
            j (int): index of a defect
        """
        d_i = self.feat["defects"][i]
        d_j = self.feat["defects"][j]
        assert d_i["match"] == d_j["match"], "defect group are not within the same target area."
        if sum([len(i) for i in d_i["polygon"]]) == len(d_i["polygon"]):
            d_i["polygon"] = [i[0] for i in d_i["polygon"]]
        if sum([len(i) for i in d_j["polygon"]]) == len(d_j["polygon"]):
            d_j["polygon"] = [i[0] for i in d_j["polygon"]]
        all_polys = []
        if len(d_i["polygon"][0]) == 2:
            all_polys.append(d_i["polygon"])
        else:
            [all_polys.append(i) for i in d_i["polygon"]]
        if len(d_j["polygon"][0]) == 2:
            all_polys.append(d_j["polygon"])
        else:
            [all_polys.append(i) for i in d_j["polygon"]]
        [self.feat["defects"].remove(self.feat["defects"][indx]) for indx in [max(i, j), min(i, j)]]
        area_sum = d_i["area"] + d_j["area"]
        # Calculating the convex hull polygon and area
        points = []
        for l in all_polys:
            assert len(l[0]) == 2, "too many points"
            [points.append(p) for p in l]
        points = np.array(points)
        hull = ConvexHull(points)
        try:
            self.feat["defects"].append({
                "match": d_i["match"],
                "polygon": all_polys,
                "area": area_sum,
                "hull": hull.volume,
                "diameter": (d_i["diameter"]/2) + (d_j["diameter"]/2) + dist,
                "x": (d_i["x"]*(d_i["area"]/area_sum)) + (d_j["x"]*(d_j["area"]/area_sum)),
                "y": (d_i["y"]*(d_i["area"]/area_sum)) + (d_j["y"]*(d_j["area"]/area_sum)),
                "distance": min(d_i["distance"], d_j["distance"]),
                "aspect": max(d_i["aspect"], d_j["aspect"]),
                "circularity": 4*math.pi*(hull.volume/(hull.area**2)),
                "roundness": (d_i["roundness"] + d_j["roundness"]) / 2,
                "solidity": float(area_sum) / hull.volume,
                "extent": (d_i["extent"] + d_j["extent"]) / 2,
                "major": max(d_i["major"], d_j["major"]),
                "minor": max(d_i["minor"], d_j["minor"])
            })
        except KeyError:
            cv2.imshow("Mask", self.mask_img)
            cv2.waitKey(0)
            raise KeyError

    def _recursive_group(self, i, j, grouped_distances=[]):
        """Recursively goes through the defects and checks if any are close enough to be grouped 
        together. Ones that pass are sent to _replace_pair where they can be merged.

        ### Args:
            i (int): an index from the defect list
            j (int): an index from the defect list
            grouped_distances (list, optional): list of distances that have already been found. Used
            to ignore duplicates. Defaults to [].

        ### Returns:
            None: returns None when complete
        """
        # checks if the iterations have finished
        if i == -1 and j == -1:
            return None
        if i == 0 and j == 0:
            return None
        d_i = self.feat["defects"][i]
        d_j = self.feat["defects"][j]
        p1 = (d_i["x"], d_i["y"])
        p2 = (d_j["x"], d_j["y"])
        distance = self.dist(p1, p2)
        # checks if the i, j combo has been seen prior
        if d_i == d_j or distance in grouped_distances or distance == 0.0:
            if j == 0:
                self._recursive_group(i-1, len(self.feat["defects"])-1, grouped_distances)
            else:
                self._recursive_group(i, j-1, grouped_distances)
        elif (distance < d_i["diameter"]):
            grouped_distances.append(distance)
            self._replace_pair(distance, i, j)
            self._recursive_group(len(self.feat["defects"])-1, len(self.feat["defects"])-1, grouped_distances)
        elif j == 0:
            self._recursive_group(i-1, len(self.feat["defects"])-1, grouped_distances)
        else:
            self._recursive_group(i, j-1, grouped_distances)                   

    def __str__(self):
        ret = \
        """(ZLayer: %i, Defect Count: %i)""" % (self.feat["z value"], len(self.feat["defects"]))
        return ret

    def __repr__(self):
        ret = \
        """(ZLayer: %i, Defect Count: %i)""" % (self.feat["z value"], len(self.feat["defects"]))
        return ret

if __name__ == "__main__":
    img_folder = "E:/reg_CT/5-1_Slices/raw/"
    areas = target_area(img_folder)
    avg_target_area = sum(areas) / len(areas)
    layer = ct_layer(
        C=3, 
        file_name="E:/reg_CT/5-1_Slices/raw/5-1_1436.tif", 
        target_area=avg_target_area,
        export_folder="C:/Users/austi/Desktop"
    )
    print("Done.")