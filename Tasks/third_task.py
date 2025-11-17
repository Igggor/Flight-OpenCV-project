import cv2
import numpy as np

IMG_PATH = "arrows.png"  # путь к изображению (черные стрелки на белом фоне)
MIN_CONTOUR_AREA = 50
LOCAL_THICKNESS_RADIUS = 5
SHOW_WINDOWS = True

# =================================
# Утончение бинарного изображения
def thinning(img):
    skel = np.zeros_like(img)
    img = img.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    while True:
        eroded = cv2.erode(img, kernel)
        temp = cv2.dilate(eroded, kernel)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break
    return skel

def local_thickness(mask, pt, radius=LOCAL_THICKNESS_RADIUS):
    x, y = int(pt[0]), int(pt[1])
    h, w = mask.shape
    x0 = max(0, x-radius)
    x1 = min(w, x+radius+1)
    y0 = max(0, y-radius)
    y1 = min(h, y+radius+1)
    patch = mask[y0:y1, x0:x1]
    return int(np.sum(patch>0))

def vector_from_skeleton(cnt, mask):
    x, y, w, h = cv2.boundingRect(cnt)
    roi_mask = mask[y:y+h, x:x+w]

    # Skeletonization
    skel = thinning(roi_mask)
    ys, xs = np.where(skel>0)
    if len(xs)<2:
        return None
    pts = np.vstack([xs, ys]).T.astype(np.float32)

    # Fit line
    [vx, vy, x0, y0] = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
    axis = np.array([vx, vy]).reshape(2)
    axis = axis / np.linalg.norm(axis)  # Нормализуем ось
    point_on_line = np.array([x0, y0]).reshape(2)

    # Проекции точек на линию
    rel_pts = pts - point_on_line
    projections = rel_pts.dot(axis)
    idx_min = np.argmin(projections)
    idx_max = np.argmax(projections)
    p_tail = pts[idx_min] + np.array([x, y])
    p_head_approx = pts[idx_max] + np.array([x, y])

    # Коррекция головы: берем крайнюю точку наконечника по минимальной толщине
    # с небольшой коррекцией для устранения диагонального смещения
    ys_roi, xs_roi = np.where(roi_mask>0)
    roi_pts = np.vstack([xs_roi, ys_roi]).T
    thicknesses = np.array([local_thickness(roi_mask, pt) for pt in roi_pts])
    min_thick = np.min(thicknesses)
    
    # Берем только пиксели кончика (минимальная толщина)
    tip_pts_mask = thicknesses == min_thick
    tip_pts = roi_pts[tip_pts_mask]
    
    if len(tip_pts)>0:
        # Берем максимальную проекцию (крайний кончик)
        tip_pts_global = tip_pts + np.array([x, y])
        rel_tip_pts = tip_pts_global - point_on_line
        tip_projections = rel_tip_pts.dot(axis)
        idx_tip_max = np.argmax(tip_projections)
        p_head_tip = tip_pts_global[idx_tip_max]
        
        # Небольшая коррекция: если есть соседние точки, усредняем с ними
        max_proj = np.max(tip_projections)
        neighbor_threshold = max_proj - 1.5  # Толщина в 1-1.5 пикселя
        neighbor_mask = tip_projections >= neighbor_threshold
        if np.sum(neighbor_mask) > 1:
            # Усредняем топ-точку с соседями для уменьшения шума
            p_head = np.mean(tip_pts_global[neighbor_mask], axis=0)
        else:
            p_head = p_head_tip
    else:
        p_head = p_head_approx

    dx = int(np.ceil(p_head[0]-p_tail[0]))
    dy = int(np.ceil(p_head[1]-p_tail[1]))

    return dx, dy, (int(p_tail[0]), int(p_tail[1])), (int(p_head[0]), int(p_head[1]))

def process_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError("Файл не найден: "+path)
    orig = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Морфология
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    sum_x, sum_y = 0, 0
    vectors = []

    for cnt in contours:
        if cv2.contourArea(cnt)<MIN_CONTOUR_AREA:
            continue
        res = vector_from_skeleton(cnt, bw)
        if res is None:
            continue
        dx, dy, tail, head = res
        sum_x += dx
        sum_y += dy
        vectors.append((dx, dy))
        cv2.arrowedLine(orig, tail, head, (0,0,255), 2, tipLength=0.2)
        cv2.circle(orig, tail, 3, (255,0,0), -1)
        cv2.circle(orig, head, 3, (0,255,0), -1)

    # print("\nВекторы (dx, dy):")
    # for i,v in enumerate(vectors,1):
    #     print(f"{i}: dx={v[0]} dy={v[1]}")
    # print(f"\nСуммарный вектор: X={sum_x}, Y={sum_y}")

    # if SHOW_WINDOWS:
    #     cv2.imshow("Result", orig)
    #     cv2.imshow("Binary", bw)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    return sum_x, sum_y, vectors


if __name__ == "__main__":
    IMG = "Examples/vec1.png"
    s_x, s_y, vcs = process_image(IMG)
    print("ANSWER", s_x + s_y)
