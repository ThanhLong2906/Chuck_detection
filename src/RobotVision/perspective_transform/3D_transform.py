import cv2
import numpy as np

# =========================
# 1. LOAD IMAGE
# =========================
img = cv2.imread("/home/long/PROJECTS/AI/Z131_Robot_Thai-Nguyen/perspective_tranform/imgs/anh_nghieng.png")
h, w = img.shape[:2]

# =========================
# 2. DETECT ELLIPSE (MIỆNG CỐC)
# =========================
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5,5), 0)
edges = cv2.Canny(blur, 50, 150)

contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)

ellipse = None
for cnt in contours:
    if len(cnt) > 100:
        ellipse = cv2.fitEllipse(cnt)
        break

if ellipse is None:
    raise Exception("Không tìm được ellipse")

(cx, cy), (MA, ma), angle = ellipse

# =========================
# 3. MODEL 3D (FRUSTUM)
# =========================
H = 7.0
r_top = 2.0
r_bot = 3.5

num_h = 50
num_theta = 80

points_3d = []

for i in range(num_h):
    z = H * i / (num_h - 1)

    # nội suy bán kính
    r = r_top + (r_bot - r_top) * (z / H)

    for j in range(num_theta):
        theta = 2*np.pi * j / num_theta

        X = r * np.cos(theta)
        Y = r * np.sin(theta)
        Z = z

        points_3d.append([X, Y, Z])

points_3d = np.array(points_3d, dtype=np.float32)

# =========================
# 4. TẠO POINTS CHO solvePnP (miệng cốc)
# =========================
obj_points = []
img_points = []

for theta in np.linspace(0, 2*np.pi, 40):
    X = r_top * np.cos(theta)
    Y = r_top * np.sin(theta)
    Z = 0
    obj_points.append([X, Y, Z])

    x = (MA/2) * np.cos(theta)
    y = (ma/2) * np.sin(theta)

    ang = np.radians(angle)
    xr = x*np.cos(ang) - y*np.sin(ang)
    yr = x*np.sin(ang) + y*np.cos(ang)

    img_points.append([cx + xr, cy + yr])

obj_points = np.array(obj_points, dtype=np.float32)
img_points = np.array(img_points, dtype=np.float32)

# =========================
# 5. CAMERA MATRIX
# =========================
f = w
K = np.array([
    [f, 0, w/2],
    [0, f, h/2],
    [0, 0, 1]
], dtype=np.float32)

dist = np.zeros((4,1))

# =========================
# 6. SOLVE PNP
# =========================
_, rvec, tvec = cv2.solvePnP(obj_points, img_points, K, dist)

R, _ = cv2.Rodrigues(rvec)

# =========================
# 7. CAMERA "THẲNG"
# =========================
R_front = np.eye(3)
t_front = tvec.copy()  # giữ nguyên khoảng cách

# =========================
# 8. RENDER ẢNH MỚI
# =========================
def warp_triangle(img_src, img_dst, tri_src, tri_dst):
    r1 = cv2.boundingRect(tri_src)
    r2 = cv2.boundingRect(tri_dst)

    # ❗ CHECK 1: rect hợp lệ
    if r1[2] <= 0 or r1[3] <= 0 or r2[2] <= 0 or r2[3] <= 0:
        return

    # ❗ CHECK 2: nằm trong ảnh
    h, w = img_src.shape[:2]
    if (r1[0] < 0 or r1[1] < 0 or
        r1[0]+r1[2] > w or
        r1[1]+r1[3] > h):
        return

    tri1_cropped = []
    tri2_cropped = []

    for i in range(3):
        tri1_cropped.append([
            tri_src[i][0] - r1[0],
            tri_src[i][1] - r1[1]
        ])
        tri2_cropped.append([
            tri_dst[i][0] - r2[0],
            tri_dst[i][1] - r2[1]
        ])

    tri1_cropped = np.float32(tri1_cropped)
    tri2_cropped = np.float32(tri2_cropped)

    img1_crop = img_src[r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]]

    # ❗ CHECK 3: crop không rỗng
    if img1_crop.size == 0:
        return

    M = cv2.getAffineTransform(tri1_cropped, tri2_cropped)

    warped = cv2.warpAffine(
        img1_crop,
        M,
        (r2[2], r2[3]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )

    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(tri2_cropped), (1,1,1), 16, 0)

    img_dst[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = \
        img_dst[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * (1 - mask) + warped * mask

output = np.zeros_like(img)

for i in range(num_h - 1):
    for j in range(num_theta):

        p1 = points_3d[i*num_theta + j]
        p2 = points_3d[i*num_theta + (j+1)%num_theta]
        p3 = points_3d[(i+1)*num_theta + j]

        tri = [p1, p2, p3]

        pts_img = []
        pts_front = []

        for pt in tri:
            X = pt.reshape(3,1)

            # project sang ảnh gốc
            Xc = R @ X + tvec
            proj = K @ Xc
            proj /= proj[2,0]
            u, v = int(proj[0,0]), int(proj[1,0])
            pts_img.append([u, v])

            # project sang front view
            Xf = X + tvec  # camera thẳng
            proj2 = K @ Xf
            proj2 /= proj2[2,0]
            u2, v2 = int(proj2[0,0]), int(proj2[1,0])
            pts_front.append([u2, v2])

        pts_img = np.array(pts_img, dtype=np.int32)
        pts_front = np.array(pts_front, dtype=np.int32)

        # lấy màu trung bình từ ảnh gốc
        mask = np.zeros((h,w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, pts_img, 255)

        mean_color = cv2.mean(img, mask=mask)[:3]

        # vẽ lên output
        warp_triangle(img, output, pts_img.astype(np.float32), pts_front.astype(np.float32))
# =========================
# 9. HIỂN THỊ
# =========================
debug = img.copy()
cv2.ellipse(debug, ellipse, (0,255,0), 2)

cv2.imshow("Original", img)
cv2.imshow("Ellipse", debug)
cv2.imshow("3D Reprojected Front View", output)

cv2.waitKey(0)
cv2.destroyAllWindows()