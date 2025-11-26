import cv2
import os
import numpy as np
import argparse
from filters import GaussianFilter
from gradient import GradientCompute


def double_threshold(img, low_ratio=0.05, high_ratio=0.15):
    
    high_thresh = img.max() * high_ratio
    low_thresh = high_thresh * low_ratio

    strong = 255
    weak = 70

    result_image = np.zeros_like(img, dtype=np.uint8)

    strong_i, strong_j = np.where(img >= high_thresh)
    weak_i, weak_j = np.where((img <= high_thresh) & (img >= low_thresh))

    result_image[strong_i, strong_j] = strong
    result_image[weak_i, weak_j] = weak

    return result_image, weak, strong


def hysteresis_recursive(img, weak, strong=255):
    #recursive thresholding
    rows, cols = img.shape

    def dfs(r, c):
        if r < 1 or r >= rows - 1 or c < 1 or c >= cols - 1:
            return
        if img[r, c] == strong:
            return
        if img[r, c] == weak:
            img[r, c] = strong
            dfs(r + 1, c); dfs(r - 1, c); dfs(r, c + 1); dfs(r, c - 1)
            dfs(r + 1, c + 1); dfs(r - 1, c - 1); dfs(r + 1, c - 1); dfs(r - 1, c + 1)

    strong_points = np.argwhere(img == strong)
    for r, c in strong_points:
        dfs(r, c)

    img[img == weak] = 0
    img[0, :] = img[-1, :] = img[:, 0] = img[:, -1] = 0  # border pixels = 0
    return img


def process_image(img_path, output_folder, sigma, T, output_ext):
    img_color = cv2.imread(img_path)
    img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    if img is None:
        print(f"no image at: {img_path}")
        return

    base_name = os.path.splitext(os.path.basename(img_path))[0]
    hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
    B = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8)
    hsv_blur = cv2.GaussianBlur(hsv, (5, 5), 0)   
    lower_yellow = np.array([15, 80, 80])
    upper_yellow = np.array([40, 255, 255])
    mask_yellow = cv2.inRange(hsv_blur, lower_yellow, upper_yellow)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 60, 255])
    mask_white = cv2.inRange(hsv_blur, lower_white, upper_white)

    mask_lane = cv2.bitwise_or(mask_white, mask_yellow)

    B_filtered = (mask_lane > 0).astype(np.uint8)

    #save for visualization
    cv2.imwrite(os.path.join(output_folder, f"{base_name}_lane_mask.png"),
                B_filtered * 255)

    img = cv2.bitwise_and(img, img, mask=B_filtered)

    #Gaussian filters
    gf = GaussianFilter(sigma, T)
    Gx, Gy, scale = gf.Gx, gf.Gy, gf.scale

    #Convolution
    fx = cv2.filter2D(img, cv2.CV_64F, Gx)
    fy = cv2.filter2D(img, cv2.CV_64F, Gy)

    #Save fx, fy
    cv2.imwrite(os.path.join(output_folder, f"{base_name}_fx_{sigma}.{output_ext}"),
                cv2.normalize(fx, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
    cv2.imwrite(os.path.join(output_folder, f"{base_name}_fy_{sigma}.{output_ext}"),
                cv2.normalize(fy, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))

    #Magnitude 
    M = GradientCompute.compute_magnitude(fx, fy, scale_factor=scale)
    cv2.imwrite(os.path.join(output_folder, f"{base_name}_magnitude_{sigma}.{output_ext}"), M)

    #Direction + Quantization
    theta = GradientCompute.compute_direction(fx, fy)
    theta_q = GradientCompute.quantize_direction(theta)

    theta_q_img = (theta_q * 85).astype(np.uint8)
    cv2.imwrite(os.path.join(output_folder, f"{base_name}_quantized_{sigma}.{output_ext}"), theta_q_img)

    #Non-Maximum Suppression
    nms = GradientCompute.non_maximum_suppression(M, theta_q)
    cv2.imwrite(os.path.join(output_folder, f"{base_name}_nms_{sigma}.{output_ext}"), nms)

    # Step 7: Double Thresholding
    dt, weak, strong = double_threshold(nms)
    cv2.imwrite(os.path.join(output_folder, f"{base_name}_double_threshold_{sigma}.{output_ext}"), dt)

    # Step 8: Hysteresis
    final = hysteresis_recursive(dt.copy(), weak, strong)
    cv2.imwrite(os.path.join(output_folder, f"{base_name}_final_edges_{sigma}.{output_ext}"), final)

    # Optional sigma = 1 special case
    if sigma == 1.0:
        dt1, weak1, strong1 = double_threshold(nms, low_ratio=0.05, high_ratio=0.15)
        final1 = hysteresis_recursive(dt1.copy(), weak1, strong1)
        cv2.imwrite(os.path.join(output_folder, f"{base_name}_final_edges_sigma1_set1.{output_ext}"), final1)

        dt2, weak2, strong2 = double_threshold(nms, low_ratio=0.1, high_ratio=0.2)
        final2 = hysteresis_recursive(dt2.copy(), weak2, strong2)
        cv2.imwrite(os.path.join(output_folder, f"{base_name}_final_edges_sigma1_set2.{output_ext}"), final2)

    print(f"Result_Images {base_name} (sigma={sigma}) into {output_folder}")
    #Region of Interest
    roi_edges = region_of_interest(final)
    cv2.imwrite(os.path.join(output_folder, f"{base_name}_roi_edges.png"), roi_edges)

    #Manual Hough Transform (NO built-in)
    lines, accumulator, thetas, rhos = hough_transform(roi_edges, threshold=120)

    #Draw lines
    hough_vis = draw_hough_lines(final, lines)
    cv2.imwrite(os.path.join(output_folder, f"{base_name}_hough_lines.png"), hough_vis)
    #Hough Transform
    lines, accumulator, thetas, rhos = hough_transform(roi_edges, threshold=120)

    #Filter horizontal + classify
    left_lines, right_lines = filter_horizontal_and_classify(lines)

    #Linear regression on left
    left_reg = manual_linear_regression(left_lines)
    right_reg = manual_linear_regression(right_lines)

    #Convert final edges to color for visualization
    lane_img = cv2.cvtColor(final, cv2.COLOR_GRAY2BGR)

    #Draw regression lines 
    h, w = final.shape
    y_max = h   # bottom
    y_min = int(0.60 * h)  # same as top of ROI trapezoid

    if left_reg is not None:
        m, b = left_reg
        lane_img = draw_regression_line(lane_img, m, b, y_min, y_max, (255,0,0), 5)

    if right_reg is not None:
        m, b = right_reg
        lane_img = draw_regression_line(lane_img, m, b, y_min, y_max, (0,255,0), 5)

    cv2.imwrite(os.path.join(output_folder, f"{base_name}_final_lane_lines.png"), lane_img)


def region_of_interest(edges):
    h, w = edges.shape

    #Define trapezoid points
    bottom_left   = (int(0.10*w), h)
    bottom_right  = (int(0.90*w), h)
    top_left      = (int(0.40*w), int(0.60*h))
    top_right     = (int(0.60*w), int(0.60*h))

    mask = np.zeros_like(edges)
    roi_corners = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)

    cv2.fillPoly(mask, roi_corners, 255)
    masked = cv2.bitwise_and(edges, mask)

    return masked
def hough_transform(edge_img, theta_res=1, rho_res=1, threshold=80):
    #Edge image size
    h, w = edge_img.shape

    #Theta range
    thetas = np.deg2rad(np.arange(-90, 90, theta_res))

    #Rho range
    diag_len = int(np.hypot(h, w))
    rhos = np.arange(-diag_len, diag_len, rho_res)

    #Accumulator array
    accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.uint64)
    #Edge points
    y_idxs, x_idxs = np.nonzero(edge_img)
    # Voting
    for x, y in zip(x_idxs, y_idxs):
        for t_idx, theta in enumerate(thetas):
            rho = int(x * np.cos(theta) + y * np.sin(theta))
            r_idx = rho + diag_len
            accumulator[r_idx, t_idx] += 1

    #Extract lines above a threshold
    lines = []
    for r in range(accumulator.shape[0]):
        for t in range(accumulator.shape[1]):
            if accumulator[r, t] > threshold:
                rho = rhos[r]
                theta = thetas[t]
                lines.append((rho, theta))

    return lines, accumulator, thetas, rhos
def draw_hough_lines(img, lines):
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape

    for rho, theta in lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho

        #Compute extended line for display
        x1 = int(x0 + 2000 * (-b))
        y1 = int(y0 + 2000 * (a))
        x2 = int(x0 - 2000 * (-b))
        y2 = int(y0 - 2000 * (a))

        cv2.line(img_color, (x1, y1), (x2, y2), (0,255,0), 2)

    return img_color
def hough_to_points(rho, theta, length=2000):
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho

    #Create long line points
    x1 = int(x0 + length * (-b))
    y1 = int(y0 + length * (a))
    x2 = int(x0 - length * (-b))
    y2 = int(y0 - length * (a))

    return (x1, y1, x2, y2)
def filter_horizontal_and_classify(lines, slope_threshold=0.3):
    left_lines = []
    right_lines = []

    for rho, theta in lines:
        x1, y1, x2, y2 = hough_to_points(rho, theta)

        if x2 == x1:
            continue  #avoid infinite slope

        slope = (y2 - y1) / (x2 - x1)

        #Discard near-horizontal slopes
        if abs(slope) < slope_threshold:
            continue

        #Classification by sign
        if slope < 0:
            left_lines.append((x1, y1, x2, y2))
        else:
            right_lines.append((x1, y1, x2, y2))

    return left_lines, right_lines
def manual_linear_regression(lines):
    xs = []
    ys = []

    #Collect all endpoints from all detected lines
    for x1, y1, x2, y2 in lines:
        xs.append(x1); ys.append(y1)
        xs.append(x2); ys.append(y2)

    if len(xs) < 2:
        return None 
    #least squares
    N = len(xs)
    sum_x = sum(xs)
    sum_y = sum(ys)
    sum_xy = sum([xs[i] * ys[i] for i in range(N)])
    sum_x2 = sum([x*x for x in xs])

    denominator = (N * sum_x2 - sum_x**2)
    if denominator == 0:
        return None

    m = (N * sum_xy - sum_x * sum_y) / denominator
    b = (sum_y - m * sum_x) / N

    return m, b
def draw_regression_line(img, m, b, y_min, y_max, color=(0,0,255), thickness=4):
    #Compute x for given y
    if m == 0:
        return img

    x1 = int((y_min - b) / m)
    x2 = int((y_max - b) / m)

    cv2.line(img, (x1, y_min), (x2, y_max), color, thickness)
    return img


def main():
    parser = argparse.ArgumentParser(description="Batch Gaussian Gradient Edge Detection")
    parser.add_argument("--input_folder", type=str, required=True, help="Path to input folder")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to save results")
    parser.add_argument("--input_ext", type=str, default="png", help="Input file extension (e.g. png, jpg)")
    parser.add_argument("--output_ext", type=str, default="png", help="Output file extension")

    args = parser.parse_args()

    sigma_values = [0.5, 1.0, 2.0]
    T = 0.3

    os.makedirs(args.output_folder, exist_ok=True)

    valid_extensions = [".png", ".jpg"]
    images = [f for f in os.listdir(args.input_folder) if os.path.splitext(f)[1].lower() in valid_extensions]

    if not images:
        print("No images in input folder.")
        return

    for img_file in images:
        img_path = os.path.join(args.input_folder, img_file)
        for sigma in sigma_values:
            process_image(img_path, args.output_folder, sigma, T, args.output_ext)


if __name__ == "__main__":
    main()
