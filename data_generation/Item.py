import cv2
import numpy as np
from scipy.ndimage import rotate
from math import sin, cos, radians


class Item():

    all = []

    def __init__(
            self, polygon_coords: np.ndarray[np.int32], image_path: str
    ) -> None:
        assert len(polygon_coords.shape) == 2, "The polygon coordinates " + \
            f"has incorect shape {polygon_coords.shape}. Must be of form " + \
            "(nx, 2)"
        assert polygon_coords.shape[1] == 2, "The polygon coordinate " + \
            f"has incorect shape {polygon_coords.shape}. Must be of form " + \
            "(nx, 2)"

        x1, y1 = np.min(polygon_coords[:, 0]), np.min(polygon_coords[:, 1])
        x2, y2 = np.max(polygon_coords[:, 0]), np.max(polygon_coords[:, 1])
        self.box_coords = [[x1, y1], [x2, y2]]

        self.polygon_coords_original = polygon_coords
        self.polygon_coords = self.polygon_coords_original - [x1, y1]  # <--
        self.area = self.PolyArea(polygon_coords[:, 0], polygon_coords[:, 1])

        self.image_path = image_path
        self.image = cv2.imread(image_path)[y1:y2, x1:x2]  # <--

        Item.all.append(self)
        # When rotating multiple time the same item, we want to rotate it from
        # it's original state (which is not posible with the rotate function
        # from scipy), so a copy is stored.
        self._image = self.image.copy()
        self._polygon_coords = self.polygon_coords.copy()  # <--

    @staticmethod
    def PolyArea(x, y):
        """ Return the area of a polygon using the shoelace formula """
        return np.abs(x @ np.roll(y, 1) - y @ np.roll(x, 1)) / 2

    @staticmethod
    def rotatePolygon(
        polygon: np.ndarray[np.int32], degrees: int
    ) -> np.ndarray[np.int32]:
        """ Rotate polygon the given angle about its center. """
        theta = radians(degrees)  # Convert angle to radians
        cosang, sinang = cos(theta), sin(theta)

        # find center point of Polygon to use as pivot
        n = len(polygon)
        cx = sum(p[0] for p in polygon) / n
        cy = sum(p[1] for p in polygon) / n

        new_points = []
        for p in polygon:
            x, y = p[0], p[1]
            tx, ty = x - cx, y - cy
            new_x = (tx * cosang + ty * sinang) + cx
            new_y = (-tx * sinang + ty * cosang) + cy
            new_points.append([new_x, new_y])

        new_points = np.array(new_points, np.int32)
        min_x = np.min(new_points[:, 0])
        min_y = np.min(new_points[:, 1])
        new_points -= [min_x, min_y]
        return new_points

    def rotate(self, angle: int) -> None:
        self.image = rotate(self._image, angle)
        polygon_coords = self.rotatePolygon(self._polygon_coords, angle)

        # The polygon needs to be re-centered on the cell
        h1, w1, _ = self.image.shape
        x1, y1 = np.min(polygon_coords[:, 0]), np.min(polygon_coords[:, 1])
        x2, y2 = np.max(polygon_coords[:, 0]), np.max(polygon_coords[:, 1])
        w2 = x2 - x1
        h2 = y2 - y1
        dx = int((w2 - w1) / 2)
        dy = int((h2 - h1) / 2)
        translation = dx, dy

        self.polygon_coords = polygon_coords - translation

    def get_mask(self) -> np.ndarray:
        h, w, _ = self.image.shape
        mask = np.zeros([h, w])
        cv2.fillConvexPoly(mask, self.polygon_coords, 1)
        mask = mask > 0  # To convert to Boolean
        # when rotating the item, a few pixels inside the contour can have a
        # value of zero. If we add the image as is, we have pixels with a value
        # of zero in the resulting image. To avoid that, I add another condition
        # in the mask to select only stricly positive pixels from the item.
        mask[np.sum(self.image, axis=2) == 0] = 0
        return mask

    def show(self) -> None:
        cv2.drawContours(
            self.image, [self.polygon_coords],
            0, (0, 255, 0), thickness=1
        )
        cv2.imshow('Extracted Image', self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save(self, out_path) -> None:
        cv2.imwrite(out_path, self.image)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.polygon_coords_original}, " + \
            f"{self.image_path})"


class Cell(Item):
    pass


class Bubble(Item):
    pass
