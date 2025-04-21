"""
Synthetic Sudoku image generator for on‑the‑fly training data.
Generates puzzles based on valid, solvable Sudoku grids.
"""

import random
import time
from pathlib import Path
from typing import Tuple, Optional, List

import cv2
import numpy as np
import keras

GRID_SIZE = 9
BASE_IMAGE_SIZE = 1000 # Initial rendering size before warp
CELL_SIZE = BASE_IMAGE_SIZE // GRID_SIZE
# Ensure MNIST is loaded only once
MNIST_BUCKETS: Optional[dict[int, list[np.ndarray]]] = None

# --- Sudoku Generation Helpers ---

def _is_safe(grid: np.ndarray, row: int, col: int, num: int) -> bool:
    """Check if it's safe to place 'num' at grid[row, col]."""
    # Check row
    if num in grid[row, :]:
        return False
    # Check column
    if num in grid[:, col]:
        return False
    # Check 3x3 box
    start_row, start_col = 3 * (row // 3), 3 * (col // 3)
    if num in grid[start_row:start_row + 3, start_col:start_col + 3]:
        return False
    return True

def _find_empty(grid: np.ndarray) -> Optional[Tuple[int, int]]:
    """Find an empty cell (marked with 0)."""
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if grid[r, c] == 0:
                return (r, c)
    return None

def _solve_sudoku(grid: np.ndarray) -> bool:
    """Solve the Sudoku grid in-place using backtracking."""
    find = _find_empty(grid)
    if not find:
        return True  # Solved
    else:
        row, col = find

    nums = list(range(1, GRID_SIZE + 1))
    random.shuffle(nums) # Introduce randomness for generation

    for num in nums:
        if _is_safe(grid, row, col, num):
            grid[row, col] = num
            if _solve_sudoku(grid):
                return True
            grid[row, col] = 0  # Backtrack

    return False

def _generate_sudoku_solution() -> np.ndarray:
    """Generate a complete, valid Sudoku grid."""
    while True: # Keep trying until a valid solution is found
        grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        if _solve_sudoku(grid):
            return grid
        # print("Failed to generate solution, retrying...") # Optional debug
        time.sleep(0.01) # Avoid busy-waiting if something goes wrong

def _create_puzzle(solution: np.ndarray, difficulty: float = 0.5) -> np.ndarray:
    """
    Create a puzzle by removing cells from a solution.
    Difficulty: approx. fraction of cells to remove (0.1 easy, 0.5 medium, 0.7 hard).
    Note: This simple removal doesn't guarantee unique solvability, but ensures
          the underlying pattern is valid.
    """
    puzzle = solution.copy()
    num_cells = GRID_SIZE * GRID_SIZE
    num_remove = int(num_cells * difficulty)

    attempts = 0
    removed_count = 0
    while removed_count < num_remove and attempts < num_cells * 2:
        row, col = random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)
        if puzzle[row, col] != 0:
            puzzle[row, col] = 0
            removed_count += 1
        attempts += 1
    # print(f"Removed {removed_count} cells for difficulty {difficulty}") # Optional debug
    return puzzle

# --- MNIST Loading ---

def _load_mnist_digits() -> dict[int, list[np.ndarray]]:
    """
    Download MNIST once and bucket images by label for rendering.
    """
    global MNIST_BUCKETS
    if MNIST_BUCKETS is not None:
        return MNIST_BUCKETS

    print("Loading MNIST dataset for rendering...")
    try:
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        images = np.concatenate([x_train, x_test])
        labels = np.concatenate([y_train, y_test])

        buckets: dict[int, list[np.ndarray]] = {i: [] for i in range(10)}
        for img, lbl in zip(images, labels):
            # Invert (black digit on white bg) and add padding
            inv = cv2.bitwise_not(img)
            padded = cv2.copyMakeBorder(
                inv, 4, 4, 4, 4,
                cv2.BORDER_CONSTANT, value=255 # White padding
            )
            buckets[int(lbl)].append(padded)
        print("MNIST loaded.")
        MNIST_BUCKETS = buckets
        return MNIST_BUCKETS
    except Exception as e:
        print(f"Error loading MNIST: {e}. Will only use font rendering.")
        MNIST_BUCKETS = {i: [] for i in range(10)} # Empty buckets
        return MNIST_BUCKETS


# --- Renderer Class ---

class SudokuRenderer:
    """
    Render a random Sudoku puzzle (based on a valid solution) to a synthetic image.
    """

    def __init__(self) -> None:
        self.mnist_buckets = _load_mnist_digits() # Ensure MNIST is loaded/attempted

    def _digit_source(self, digit: int) -> Tuple[Optional[np.ndarray], str]:
        """Choose between MNIST digit or font rendering."""
        sources = []
        # Only allow MNIST if buckets[digit] is not empty
        if self.mnist_buckets and self.mnist_buckets.get(digit):
            sources.append("mnist")
        sources.append("font") # Always allow font as fallback
        choice = random.choice(sources)

        if choice == "mnist":
            img = random.choice(self.mnist_buckets[digit])
            # Ensure it's BGR for consistency
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            return img, "mnist"
        # Fallback to font rendering if MNIST chosen but failed, or if font chosen
        return None, "font"

    def render_sudoku(
        self,
        grid_spec: Optional[np.ndarray] = None,
        *,
        allow_empty: bool = True, # If True, difficulty is random, otherwise uses grid_spec or full solution
        difficulty: float = 0.5 # Used if allow_empty=True and grid_spec=None
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Generate a synthetic Sudoku image based on a valid grid.

        Args:
            grid_spec: A specific (9, 9) puzzle grid (0 for empty). If None, generates randomly.
            allow_empty: If True and grid_spec is None, generate a puzzle with random difficulty.
                         If False and grid_spec is None, render the full solution.
            difficulty: Target fraction of empty cells if generating randomly (approx).

        Returns:
            Tuple of (image, ground_truth_puzzle_grid, warped_corners).
            Returns (None, None, None) on generation failure.
        """
        try:
            if grid_spec is not None:
                # Use the provided puzzle spec directly
                gt_puzzle = grid_spec.copy()
                # We don't have the full solution here, but the puzzle itself is the ground truth
            else:
                # Generate a full solution first
                solution = _generate_sudoku_solution()
                if not allow_empty:
                    # Render the full solution
                    gt_puzzle = solution.copy()
                else:
                    # Create a puzzle by removing digits from the solution
                    rand_difficulty = random.uniform(0.3, 0.7) # Randomize difficulty
                    gt_puzzle = _create_puzzle(solution, difficulty=rand_difficulty)

            # --- Start Rendering ---
            bg_color = tuple(random.randint(200, 240) for _ in range(3))
            img = np.full((BASE_IMAGE_SIZE, BASE_IMAGE_SIZE, 3), bg_color, np.uint8)

            # Draw grid lines
            for i in range(GRID_SIZE + 1):
                major = (i % 3 == 0)
                thickness = random.randint(3 if major else 1, 5 if major else 3)
                color = (0, 0, 0) # Black lines
                # Horizontal lines
                cv2.line(img, (0, i * CELL_SIZE), (BASE_IMAGE_SIZE, i * CELL_SIZE), color, thickness)
                # Vertical lines
                cv2.line(img, (i * CELL_SIZE, 0), (i * CELL_SIZE, BASE_IMAGE_SIZE), color, thickness)

            # Draw digits onto the grid based on gt_puzzle
            for r in range(GRID_SIZE):
                for c in range(GRID_SIZE):
                    digit_to_render = gt_puzzle[r, c]
                    if digit_to_render == 0:
                        continue # Skip empty cells

                    src_img, src_type = self._digit_source(digit_to_render)
                    scale = random.uniform(0.5, 0.8) # Size relative to cell
                    target_size = int(CELL_SIZE * scale)
                    if target_size < 10: continue # Skip if too small

                    # Calculate center position with jitter
                    center_x = c * CELL_SIZE + CELL_SIZE // 2
                    center_y = r * CELL_SIZE + CELL_SIZE // 2
                    dx = int(random.uniform(-0.1, 0.1) * CELL_SIZE)
                    dy = int(random.uniform(-0.1, 0.1) * CELL_SIZE)
                    cx, cy = center_x + dx, center_y + dy

                    if src_type == "mnist" and src_img is not None:
                        # Resize MNIST digit
                        digit_img = cv2.resize(src_img, (target_size, target_size), interpolation=cv2.INTER_AREA)

                        # Random rotation
                        angle = random.uniform(-10, 10)
                        M = cv2.getRotationMatrix2D((target_size / 2, target_size / 2), angle, 1)
                        digit_img = cv2.warpAffine(
                            digit_img, M, (target_size, target_size),
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255) # Match padding
                        )

                        # Create mask (assuming black digit on white background from MNIST processing)
                        mask = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
                        _, mask = cv2.threshold(mask, 250, 255, cv2.THRESH_BINARY_INV) # Invert to get digit mask

                        # Calculate ROI in the main image
                        x0 = max(0, cx - target_size // 2)
                        y0 = max(0, cy - target_size // 2)
                        x1 = min(img.shape[1], x0 + target_size)
                        y1 = min(img.shape[0], y0 + target_size)
                        roi = img[y0:y1, x0:x1]

                        # Adjust digit image and mask if ROI is smaller (near edges)
                        digit_roi = digit_img[0:roi.shape[0], 0:roi.shape[1]]
                        mask_roi = mask[0:roi.shape[0], 0:roi.shape[1]]
                        mask_inv_roi = cv2.bitwise_not(mask_roi)

                        # Place digit using mask
                        bg_region = cv2.bitwise_and(roi, roi, mask=mask_inv_roi)
                        fg_region = cv2.bitwise_and(digit_roi, digit_roi, mask=mask_roi)
                        img[y0:y1, x0:x1] = cv2.add(bg_region, fg_region)

                    else: # Use font rendering
                        font = cv2.FONT_HERSHEY_SIMPLEX # Or try FONT_HERSHEY_DUPLEX etc.
                        thickness = random.randint(2, 4)
                        # Adjust font scale to fit target size
                        font_scale = cv2.getFontScaleFromHeight(font, target_size, thickness) * 0.8
                        text = str(digit_to_render)
                        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)

                        # Calculate position for font rendering
                        text_x = cx - tw // 2
                        text_y = cy + th // 2
                        cv2.putText(img, text, (text_x, text_y),
                                    font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

            # --- Post-processing ---
            # Add noise
            noise_level = random.uniform(5, 15)
            noise = np.random.normal(0, noise_level, img.shape).astype(np.float32)
            noisy_img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

            # Random perspective warp
            h, w = noisy_img.shape[:2]
            orig_corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype="float32")
            shift_factor = random.uniform(0.02, 0.15) # Reduced max shift slightly
            max_dx, max_dy = w * shift_factor, h * shift_factor

            # Generate new corner positions with some randomness
            warped_corners = np.array([
                [random.uniform(0, max_dx), random.uniform(0, max_dy)], # Top-left
                [w - 1 - random.uniform(0, max_dx), random.uniform(0, max_dy)], # Top-right
                [w - 1 - random.uniform(0, max_dx), h - 1 - random.uniform(0, max_dy)], # Bottom-right
                [random.uniform(0, max_dx), h - 1 - random.uniform(0, max_dy)], # Bottom-left
            ], dtype="float32")

            # Ensure corners maintain roughly the correct order (prevent extreme warps)
            # Simple check: ensure TL x < TR x, BL x < BR x, TL y < BL y, TR y < BR y
            if (warped_corners[0,0] >= warped_corners[1,0] or \
                warped_corners[3,0] >= warped_corners[2,0] or \
                warped_corners[0,1] >= warped_corners[3,1] or \
                warped_corners[1,1] >= warped_corners[2,1]):
                 # If order is messed up, use less aggressive warp or skip warp
                 warped_corners = orig_corners # Fallback to no warp this time

            M = cv2.getPerspectiveTransform(orig_corners, warped_corners)

            # Calculate output bounds for warped image
            out_w = int(np.max(warped_corners[:, 0])) + 1
            out_h = int(np.max(warped_corners[:, 1])) + 1
            out_w = max(out_w, 100) # Ensure minimum size
            out_h = max(out_h, 100)

            warped_img = cv2.warpPerspective(
                noisy_img, M, (out_w, out_h),
                flags=cv2.INTER_LINEAR, # Smoother interpolation
                borderMode=cv2.BORDER_REPLICATE # Replicate border pixels
            )

            # Apply slight blur after warping
            final_img = cv2.GaussianBlur(warped_img, (3, 3), 0)

            # Return the final image, the puzzle grid (0 for empty), and the warped corners
            return final_img, gt_puzzle, warped_corners

        except Exception as e:
            print(f"[Renderer Error] Failed to render Sudoku: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None


def generate_and_save_test_example(
    prefix: str = "epoch_test_sudoku",
    force: bool = False
) -> Tuple[Optional[str], Optional[np.ndarray]]:
    """
    Generate or load a fixed Sudoku test example for epoch callbacks.
    Returns (image_path, ground_truth_puzzle_grid).
    """
    img_path = Path(f"{prefix}.png")
    gt_path = Path(f"{prefix}_gt.npy")

    if not force and img_path.exists() and gt_path.exists():
        try:
            gt = np.load(gt_path)
            # Basic check if loaded files are valid
            if cv2.imread(str(img_path)) is not None and gt.shape == (GRID_SIZE, GRID_SIZE):
                 print(f"Loaded existing test example: {img_path}, {gt_path}")
                 return str(img_path), gt
            else:
                 print("Existing test example files corrupted, regenerating...")
        except Exception as e:
            print(f"Error loading existing test example ({e}), regenerating...")

    print("Generating new test example...")
    renderer = SudokuRenderer()
    # Define a specific puzzle grid (0 for empty)
    # This puzzle should ideally be solvable and have a unique solution
    # Example puzzle (source: websudoku.com easy)
    test_puzzle_grid = np.array([
        [0, 0, 3, 0, 2, 0, 6, 0, 0],
        [9, 0, 0, 3, 0, 5, 0, 0, 1],
        [0, 0, 1, 8, 0, 6, 4, 0, 0],
        [0, 0, 8, 1, 0, 2, 9, 0, 0],
        [7, 0, 0, 0, 0, 0, 0, 0, 8],
        [0, 0, 6, 7, 0, 8, 2, 0, 0],
        [0, 0, 2, 6, 0, 9, 5, 0, 0],
        [8, 0, 0, 2, 0, 3, 0, 0, 9],
        [0, 0, 5, 0, 1, 0, 3, 0, 0]
    ], dtype=int)

    # Render this specific puzzle
    img, gt, _ = renderer.render_sudoku(grid_spec=test_puzzle_grid)

    if img is None or gt is None:
        print("[Error] Failed to generate test example image.")
        return None, None

    try:
        cv2.imwrite(str(img_path), img)
        np.save(gt_path, gt)
        print(f"Saved new test example: {img_path}, {gt_path}")
        return str(img_path), gt
    except Exception as e:
        print(f"[Error] Failed to save test example: {e}")
        return None, None

# Example usage for debugging
if __name__ == "__main__":
    print("Generating a sample Sudoku image...")
    renderer = SudokuRenderer()
    img, gt, corners = renderer.render_sudoku(allow_empty=True, difficulty=0.6)

    if img is not None:
        print("Generated Ground Truth Puzzle Grid:")
        print(gt)
        cv2.imwrite("sample_sudoku_generated.png", img)
        print("Saved sample image to sample_sudoku_generated.png")

        # Also generate the fixed test example if needed
        generate_and_save_test_example(force=True)
    else:
        print("Failed to generate sample image.")
