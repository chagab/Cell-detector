import os
import cv2
import numpy as np
from Item import Item, Cell, Bubble
import random


class MockImage():

    all = []

    def __init__(
            self, background_path: str, mean_angle: float, confluence: int,
            trials: int, mock_image_path: str, output_dir_image='images',
            output_dir_label='labels', range_bubbles=(1, 5)
    ) -> None:
        self.background_path = background_path
        self.mean_angle = mean_angle
        self.confluence = confluence
        self.trials = trials
        self.range_bubbles = range_bubbles
        self.labels = []

        self.image = cv2.imread(background_path)
        h, w, _ = self.image.shape
        self.segmentation_image = np.zeros([h, w])

        MockImage.all.append(self)
        self.id = len(MockImage.all)
        confluence_str = f'{self.confluence}'.replace('.', '_')
        mock_image_name = os.path.split(mock_image_path)[-1]
        self.image_path = f'{output_dir_image}\{mock_image_name}-confluence{confluence_str}-{self.id:03d}.tif'
        self.label_path = f'{output_dir_label}\{mock_image_name}-confluence{confluence_str}-{self.id:03d}.txt'

    def add_bubbles(
        self, items: list[Item], range_items: tuple[int], trials=5,
        save_labels=False
    ) -> None:
        h, w, _ = self.image.shape
        std = 100 / self.confluence
        for id in range(random.randint(*range_items)):
            item = random.choice(items)
            angle = np.random.normal(loc=self.mean_angle, scale=std, size=None)
            item.rotate(angle)
            bh, bw, _ = item.image.shape
            for _ in range(trials):
                x, y = random.randint(0, w - bw), random.randint(0, h - bh)
                item_coords_in_image = item.polygon_coords + [x, y]
                mask = item.get_mask()
                if np.sum(self.segmentation_image[y:y+bh, x:x+bw][mask]) == 0:
                    # if item is not overlapping with anything
                    self.image[y:y+bh, x:x+bw][mask] = item.image[mask]
                    self.segmentation_image[y:y+bh, x:x+bw][mask] = 255
                    if save_labels:
                        self.labels.append({
                            'id': id,
                            'coords': item_coords_in_image
                        })
                    break

    def add_cells(self, items: list[Item]) -> None:
        id = 0
        confluence = 0
        h, w, _ = self.image.shape
        tot_pixels = h * w
        cell_pixels = 0
        std = 100 / self.confluence
        while confluence < self.confluence:
            item = random.choice(items)
            angle = np.random.normal(loc=self.mean_angle, scale=std, size=None)
            item.rotate(angle)
            bh, bw, _ = item.image.shape
            x, y = random.randint(0, w - bw), random.randint(0, h - bh)
            item_coords_in_image = item.polygon_coords + [x, y]
            mask = item.get_mask()
            area = self.segmentation_image[y:y+bh, x:x+bw][mask]
            if np.sum(area) == 0:
                # if the item is not overlapping with anything else
                self.image[y:y+bh, x:x+bw][mask] = item.image[mask]
                self.segmentation_image[y:y+bh, x:x+bw][mask] = 1
                self.labels.append({
                    'id': id,
                    'coords': item_coords_in_image
                })
                cell_pixels += len(area)
                confluence = cell_pixels / tot_pixels
                id += 1

    def generate_new_image(
            self, cells: list[Cell], bubbles: list[Bubble]
    ) -> None:
        self.add_bubbles(bubbles, range_items=self.range_bubbles)
        self.add_cells(cells)

    def show(self, contour=False) -> None:
        if contour:
            for label in self.labels:
                cv2.drawContours(
                    self.image, [label['coords']],
                    0, (0, 0, 255), thickness=2
                )
        cv2.imshow('Mock Image', self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save(self) -> None:
        h, w, _ = self.image.shape
        cv2.imwrite(self.image_path, self.image)
        with open(self.label_path, "w") as output:
            for label in self.labels:
                object_info = '0'
                for x, y in label['coords']:
                    object_info += f' {x/w:.6f} {y/h:.6f}'
                object_info += '\n'
                output.write(object_info)

    def __repr__(self) -> str:
        pass
