"""
Camera Calibration using Checkerboard Pattern
----------------------------------------------
Usage:
    1. Capture mode:  python camera_calibration.py --capture
    2. Calibrate:     python camera_calibration.py --calibrate
    3. Undistort live: python camera_calibration.py --undistort

Requirements:
    pip install opencv-python numpy matplotlib
"""

import cv2
import numpy as np
import os
import glob
import argparse
import json

# os.environ["QT_QPA_PLATFORM"] = "xcb"

# ── Configuration ────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))

CHECKERBOARD = (3, 4)          # Inner corners (columns, rows)
SQUARE_SIZE  = 2.0            # Real-world size of one square in mm (or any unit)
IMAGES_DIR   = os.path.join(SCRIPT_DIR, "calib_images")  # Where captured frames are saved
OUTPUT_FILE  = os.path.join(SCRIPT_DIR, "calibration1.json")
CONTRAST     = 1.0            # Contrast multiplier (1.0 = no change)
BRIGHTNESS   = 0              # Brightness offset (0 = no change)

# ─────────────────────────────────────────────────────────────────────────────


def capture_images(n_images: int = 20, camera_index: int = 0) -> None:
    """Interactively capture calibration frames from the live camera."""
    os.makedirs(IMAGES_DIR, exist_ok=True)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {camera_index}")

    print(f"\n[CAPTURE]  Press SPACE to save a frame ({n_images} needed), Q to quit.\n")
    count = 0

    while count < n_images:
        ret, frame = cap.read()
        if not ret:
            print("Warning: failed to grab frame, retrying…")
            continue

        if CONTRAST != 1.0 or BRIGHTNESS != 0:
            frame = cv2.convertScaleAbs(frame, alpha=CONTRAST, beta=BRIGHTNESS)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Fast path: detect on a half-resolution copy to cut search cost by ~4×.
        # CALIB_CB_FAST_CHECK pre-screens the frame cheaply and returns immediately
        # when no board-like structure is found, eliminating most of the lag.
        small = cv2.resize(gray, None, fx=0.5, fy=0.5)
        found, corners = cv2.findChessboardCorners(
            small, CHECKERBOARD,
            cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK
        )
        if found:
            # Scale corners back to full resolution and refine there
            corners = corners * 2.0
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        display = frame.copy()
        if found:
            cv2.drawChessboardCorners(display, CHECKERBOARD, corners, found)
            cv2.putText(display, "Checkerboard FOUND – press SPACE to capture",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(display, "No checkerboard detected",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(display, f"Saved: {count}/{n_images}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow("Camera Calibration – Capture", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Capture aborted by user.")
            break
        if key == ord(' ') and found:
            path = os.path.join(IMAGES_DIR, f"frame_{count:03d}.png")
            cv2.imwrite(path, frame)
            print(f"  Saved {path}")
            count += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nCapture complete: {count} images saved to '{IMAGES_DIR}/'")


def plot_coverage_map(
    img_points: list,
    image_size: tuple,
    checkerboard: tuple,
    save_path: str | None = None,
) -> None:
    """
    Draw all detected checkerboard positions onto a blank sensor canvas.

    Each frame gets a distinct colour.  For every frame we draw:
      • The four outer-corner quad (the bounding rectangle of the board)
      • All individual inner-corner dots

    Parameters
    ----------
    img_points   : list of (N_corners, 1, 2) float32 arrays, one per valid frame
    image_size   : (width, height) in pixels – the sensor size
    checkerboard : (cols, rows) inner-corner count
    save_path    : if given, save the figure to this path instead of showing it
    """
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.colors import hsv_to_rgb
    except ImportError:
        print("\n[WARNING] matplotlib not installed – skipping coverage map.\n"
              "          Install with:  pip install matplotlib\n")
        return

    n_frames = len(img_points)
    W, H = image_size

    fig, ax = plt.subplots(figsize=(10, 10 * H / W), dpi=110)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)          # Flip Y so (0,0) is top-left, matching OpenCV
    ax.set_aspect("equal")
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    ax.set_title(
        f"Calibration coverage map  ·  {n_frames} valid frames  ·  "
        f"sensor {W}×{H} px",
        color="black", fontsize=12, pad=10,
    )
    ax.set_xlabel("x  (px)", color="#888", fontsize=9)
    ax.set_ylabel("y  (px)", color="#888", fontsize=9)
    ax.tick_params(colors="#555")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

    # Sensor boundary
    ax.add_patch(mpatches.Rectangle(
        (0, 0), W, H,
        linewidth=1.2, edgecolor="#444", facecolor="none", linestyle="--",
    ))

    # Generate one hue-shifted colour per frame
    hues = np.linspace(0, 1, n_frames, endpoint=False)

    legend_handles = []
    for i, (corners, hue) in enumerate(zip(img_points, hues)):
        color = hsv_to_rgb([hue, 0.85, 0.95])

        pts = corners.reshape(-1, 2)        # (N_corners, 2)

        # ── All individual corners ───────────────────────────────────────────
        ax.scatter(
            pts[:, 0], pts[:, 1],
            s=4, color=color, alpha=0.55, linewidths=0,
        )

        # ── Board outline: convex hull of the detected corners ───────────────
        # Use the four extreme corners of the refined grid for a cleaner quad
        cols, rows = checkerboard
        # Corners are stored row-major: top-left, top-right, bottom-left, bottom-right
        tl = pts[0]
        tr = pts[cols - 1]
        bl = pts[(rows - 1) * cols]
        br = pts[rows * cols - 1]
        quad = np.array([tl, tr, br, bl, tl])          # closed polygon

        line, = ax.plot(
            quad[:, 0], quad[:, 1],
            color=color, linewidth=1.0, alpha=0.8,
        )

        # ── Frame label at board centre ──────────────────────────────────────
        cx, cy = pts.mean(axis=0)
        ax.text(
            cx, cy, str(i),
            color=color, fontsize=6, ha="center", va="center", alpha=0.7,
            fontweight="bold",
        )

        legend_handles.append(
            mpatches.Patch(color=color, label=f"frame {i:02d}")
        )

    # ── Image-centre crosshair ────────────────────────────────────────────────
    ax.axhline(H / 2, color="#555", linewidth=0.5, linestyle=":")
    ax.axvline(W / 2, color="#555", linewidth=0.5, linestyle=":")
    ax.text(W / 2 + 8, H / 2 - 8, "centre", color="#555", fontsize=7)

    # ── Legend (small, outside plot if many frames) ───────────────────────────
    ncol = max(1, n_frames // 10)
    ax.legend(
        handles=legend_handles,
        loc="lower right", fontsize=6, ncol=ncol,
        framealpha=0.9, labelcolor="black",
        facecolor="white", edgecolor="#333",
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  Coverage map saved to: {save_path}")
    else:
        plt.show()

    plt.close(fig)


def calibrate(visualise: bool = True) -> dict:
    """Run calibration on saved images and write results to JSON."""

    # Prepare 3-D object points for one board view
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0],
                            0:CHECKERBOARD[1]].T.reshape(-1, 2) * SQUARE_SIZE

    obj_points = []   # 3-D world points across all views
    img_points = []   # 2-D image points across all views

    images = sorted(glob.glob(os.path.join(IMAGES_DIR, "*.png")) +
                    glob.glob(os.path.join(IMAGES_DIR, "*.jpg")))

    if not images:
        raise FileNotFoundError(
            f"No images found in '{IMAGES_DIR}/'. Run --capture first.")

    print(f"\n[CALIBRATE]  Processing {len(images)} images …\n")
    image_size = None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)

    for path in images:
        img  = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]   # (width, height)

        found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
        if not found:
            print(f"  SKIP  {path}  (checkerboard not detected)")
            continue

        # Sub-pixel refinement
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        obj_points.append(objp)
        img_points.append(corners2)

        if visualise:
            vis = img.copy()
            cv2.drawChessboardCorners(vis, CHECKERBOARD, corners2, found)
            cv2.imshow("Calibration frames", vis)
            cv2.waitKey(200)

        print(f"  OK    {path}")

    if visualise:
        cv2.destroyAllWindows()

    if len(obj_points) < 4:
        raise RuntimeError(
            "Not enough valid frames (need ≥ 4). Capture more images or "
            "ensure the full checkerboard is visible and well-lit.")

    print(f"\nUsing {len(obj_points)} valid frame(s) for calibration…")

    # flags = (cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 |
    #      cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_TANGENT_DIST |
    #      cv2.CALIB_FIX_PRINCIPAL_POINT)
    # rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    #     obj_points, img_points, image_size, None, None, flags=flags)
    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None)

    # ── Compute per-image reprojection error ──────────────────────────────
    errors = []
    for i, (objp_i, imgp_i) in enumerate(zip(obj_points, img_points)):
        proj, _ = cv2.projectPoints(objp_i, rvecs[i], tvecs[i],
                                    camera_matrix, dist_coeffs)
        # OpenCV may return detected and projected points with different
        # channel layouts (N x 2 vs N x 1 x 2).  Normalize both before
        # comparing them; cv2.norm requires identical element types/shapes.
        imgp_flat = np.asarray(imgp_i, dtype=np.float32).reshape(-1, 2)
        proj_flat = np.asarray(proj, dtype=np.float32).reshape(-1, 2)
        errors.append(cv2.norm(imgp_flat, proj_flat, cv2.NORM_L2) /
                      len(proj_flat))
    mean_error = float(np.mean(errors))

    results = {
        "rms_reprojection_error_px": round(rms, 4),
        "mean_reprojection_error_px": round(mean_error, 4),
        "image_size_wh": list(image_size),
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.tolist(),
        "checkerboard": list(CHECKERBOARD),
        "square_size_mm": SQUARE_SIZE,
        "n_frames_used": len(obj_points),
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*55}")
    print(f"  RMS reprojection error : {rms:.4f} px")
    print(f"  Mean per-frame error   : {mean_error:.4f} px")
    print(f"  Frames used            : {len(obj_points)}")
    print(f"  Results saved to       : {OUTPUT_FILE}")
    print(f"{'='*55}\n")

    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    print(f"  Focal length  : fx={fx:.1f}  fy={fy:.1f} px")
    print(f"  Principal pt  : cx={cx:.1f}  cy={cy:.1f} px")
    print(f"  Distortion    : {dist_coeffs.ravel().round(5).tolist()}\n")

    # ── Coverage map ──────────────────────────────────────────────────────
    coverage_map_path = os.path.join(SCRIPT_DIR, "coverage_map.png")
    print("[COVERAGE MAP]  Plotting detected board positions…")
    plot_coverage_map(
        img_points=img_points,
        image_size=image_size,
        checkerboard=CHECKERBOARD,
        save_path=coverage_map_path,   # set to None to show interactively
    )

    return results


def undistort_live(calib_file: str = OUTPUT_FILE, camera_index: int = 0) -> None:
    """Show a side-by-side live preview of distorted vs undistorted feed."""
    if not os.path.exists(calib_file):
        raise FileNotFoundError(
            f"'{calib_file}' not found. Run --calibrate first.")

    with open(calib_file) as f:
        data = json.load(f)

    K    = np.array(data["camera_matrix"])
    dist = np.array(data["dist_coeffs"])
    w, h = data["image_size_wh"]

    new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
    map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, new_K, (w, h),
                                              cv2.CV_16SC2)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {camera_index}")

    print("\n[UNDISTORT]  Showing live preview. Press Q to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        if CONTRAST != 1.0 or BRIGHTNESS != 0:
            frame = cv2.convertScaleAbs(frame, alpha=CONTRAST, beta=BRIGHTNESS)

        undistorted = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)

        # Crop to valid ROI
        x, y, rw, rh = roi
        undistorted_cropped = undistorted[y:y+rh, x:x+rw]
        undistorted_resized  = cv2.resize(undistorted_cropped, (frame.shape[1], frame.shape[0]))

        combined = np.hstack([frame, undistorted_resized])
        cv2.putText(combined, "Original", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(combined, "Undistorted", (frame.shape[1] + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Undistort Preview (Q to quit)", combined)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Camera calibration with a checkerboard pattern.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--capture",   action="store_true",
                       help="Capture calibration images from the camera.")
    group.add_argument("--calibrate", action="store_true",
                       help="Run calibration on captured images.")
    group.add_argument("--undistort", action="store_true",
                       help="Show live undistorted preview using saved calibration.")
    parser.add_argument("--n-images", type=int, default=20,
                        help="Number of images to capture (default: 20).")
    parser.add_argument("--no-vis",   action="store_true",
                        help="Skip visualisation during calibration.")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera index to use (default: 0).")
    args = parser.parse_args()

    if args.capture:
        capture_images(n_images=args.n_images, camera_index=args.camera)
    elif args.calibrate:
        calibrate(visualise=not args.no_vis)
    elif args.undistort:
        undistort_live(camera_index=args.camera)


if __name__ == "__main__":
    main()