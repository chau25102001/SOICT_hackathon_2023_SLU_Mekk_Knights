from shapely.geometry import Polygon

polygon1 = Polygon([(0, 0), (1, 1), (1, 0)])
polygon2 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
intersect = polygon1.intersection(polygon2).area
union = polygon1.union(polygon2).area
iou = intersect / union
print(iou)  # iou = 0.5