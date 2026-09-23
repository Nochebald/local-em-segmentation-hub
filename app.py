import os
import shutil
import time
import colorsys
import urllib.request
import concurrent.futures
import cv2
import gradio as gr
import numpy as np
import segmentation_models_pytorch as smp
import torch
import plotly.graph_objects as go
from skimage import measure
import scipy.ndimage as ndimage
from segment_anything import SamPredictor, sam_model_registry
from torch.utils.data import DataLoader, Dataset


def resolve_file_path(file_obj):
    """Resolve Gradio file objects or plain paths to a filesystem path."""
    if file_obj is None:
        return None
    if isinstance(file_obj, str):
        return file_obj
    for attr in ("name", "path", "filepath"):
        value = getattr(file_obj, attr, None)
        if isinstance(value, str) and value:
            return value
    if isinstance(file_obj, dict):
        for key in ("name", "path", "filepath"):
            value = file_obj.get(key)
            if isinstance(value, str) and value:
                return value
    return str(file_obj)


def to_binary_mask(arr):
    """Convert grayscale/bool arrays to a boolean mask."""
    if arr is None:
        return None
    arr = np.asarray(arr)
    if arr.dtype == np.bool_:
        return arr
    if arr.max() <= 1:
        return arr > 0.5
    return arr > 127


_DEFAULT_GAP_REPAIR_KERNEL = 15  # starting closing kernel for bridging breaks in a ring's outline
_MAX_GAP_REPAIR_KERNEL = 61      # hard safety cap for adaptive escalation (see below)
_MAX_GAP_REPAIR_TRIES = 4

# Fixed & deliberately tiny - this ONLY bridges single-pixel noise/antialiasing
# gaps so a ring fragmented by multiple wall breaks still gets identified as one
# object. It must NOT scale with the user's gap-repair kernel: testing shows
# that even a modest 5px closing here is enough to bridge the ~2-4px gaps
# that commonly separate real, distinct, densely-packed myelin sheaths - and
# because this runs on the whole image at once, that single false bridge
# cascades transitively through an entire packed region, silently fusing
# dozens of separate axons into one giant blob (which is also exactly what
# produces the "everything renders in one color" symptom, since there's then
# only one object left to color). The actual per-object gap-repair below is
# free to use a much larger kernel safely, because it's clipped to each
# object's own Voronoi cell afterwards and can never cross into a neighbor.
_OBJECT_SEPARATION_KERNEL = 3


def _fill_single_component(component_u8, close_kernel):
    """One closing + flood-fill pass at a fixed kernel size, on ONE isolated
    component (every other component already zeroed out of this array), so
    the closing/fill can only ever affect this object's own shape - never a
    neighbor's."""
    h, w = component_u8.shape
    close_kernel = max(1, int(close_kernel))
    pad = max(2, close_kernel // 2 + 2)
    padded = np.pad(component_u8, pad, mode='constant', constant_values=0)

    if close_kernel > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
        closed = cv2.morphologyEx(padded, cv2.MORPH_CLOSE, kernel)
    else:
        closed = padded

    # Robust hole fill: flood-fill from the border, then invert to get only the
    # pixels that are neither foreground nor reachable background (= enclosed holes).
    ph, pw = closed.shape
    flood = closed.copy()
    ff_mask = np.zeros((ph + 2, pw + 2), np.uint8)
    cv2.floodFill(flood, ff_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    filled = cv2.bitwise_or(closed, holes)

    return filled[pad:pad + h, pad:pad + w] > 127


def _fill_single_component_adaptive(component_u8, close_kernel):
    """Fill one isolated component's interior, automatically growing the
    closing kernel if the requested size wasn't big enough to bridge a real
    break in the object's own outline - the classic "still hollow" failure
    mode, which otherwise silently depends on the break happening to be
    smaller than whatever kernel size the user (or a hardcoded constant)
    picked. A convex-hull test tells us whether this shape even *has* a
    cavity worth chasing, so solid/already-filled objects take the fast path
    and never pay for extra retries.
    """
    raw = component_u8 > 0
    raw_area = int(raw.sum())
    if raw_area == 0:
        return raw

    ys, xs = np.nonzero(raw)
    pts = np.column_stack([xs, ys]).astype(np.float32)
    hull_area = raw_area
    if len(pts) >= 3:
        hull = cv2.convexHull(pts)
        hull_area = max(raw_area, cv2.contourArea(hull))

    k = max(1, int(close_kernel))
    filled = _fill_single_component(component_u8, close_kernel=k)
    # Only worth chasing a bigger kernel if there's a real gap between this
    # shape's own area and what its convex hull suggests it "should" enclose.
    target = raw_area + 0.5 * (hull_area - raw_area)

    tries = 1
    while filled.sum() < target and k < _MAX_GAP_REPAIR_KERNEL and tries < _MAX_GAP_REPAIR_TRIES:
        k = min(_MAX_GAP_REPAIR_KERNEL, int(k * 1.7) | 1)
        filled = _fill_single_component(component_u8, close_kernel=k)
        tries += 1

    return filled


def solidify_mask_slice(mask_img, close_kernel=_DEFAULT_GAP_REPAIR_KERNEL):
    """Fill a single 2D mask completely so ring-like structures (e.g. myelin sheath
    cross-sections) become solid, opaque blobs instead of hollow rings - WITHOUT
    merging separate nearby axons into one another.

    This is object-aware: it labels distinct components first (with a tiny,
    fixed kernel that only bridges single-pixel noise, never genuinely separate
    structures - see _OBJECT_SEPARATION_KERNEL), then fills each component's
    interior in isolation (adaptively growing the kernel if needed to close a
    real wall gap), then clips every component's fill to its own Voronoi cell
    (nearest-component ownership over the whole image, including the
    background between objects). That clip is what lets an isolated object use
    a large repair kernel safely: the fill can grow well beyond the object's
    original footprint while chasing a big gap, but it can never cross into
    territory that's actually closer to a neighboring object, no matter how
    large the kernel gets.
    """
    mask = to_binary_mask(mask_img)
    if mask is None:
        return None
    if not np.any(mask):
        return np.zeros_like(mask, dtype=np.uint8)

    h, w = mask.shape
    mask_u8 = mask.astype(np.uint8)

    if _OBJECT_SEPARATION_KERNEL > 1:
        lk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_OBJECT_SEPARATION_KERNEL, _OBJECT_SEPARATION_KERNEL))
        label_source = cv2.morphologyEx(mask_u8 * 255, cv2.MORPH_CLOSE, lk)
    else:
        label_source = mask_u8 * 255
    num_labels, label_map = cv2.connectedComponents((label_source > 0).astype(np.uint8), connectivity=8)

    if num_labels <= 1:
        return np.zeros((h, w), dtype=np.uint8)

    # Voronoi ownership: for every pixel - including a ring's own interior hole
    # AND the background between two different rings - find which component's
    # nearest original pixel it belongs to. A component's fill can never cross
    # into territory that's actually closer to a different neighboring component.
    bg = label_map == 0
    _, indices = ndimage.distance_transform_edt(bg, return_distances=True, return_indices=True)
    nearest_label = label_map[tuple(indices)]

    result = np.zeros((h, w), dtype=bool)
    for k in range(1, num_labels):
        component_mask = label_map == k
        component = component_mask.astype(np.uint8) * 255
        filled_component = _fill_single_component_adaptive(component, close_kernel=close_kernel)
        # Erode ownership by 1px so a guaranteed background pixel survives
        # between any two neighboring objects that share a Voronoi border -
        # without this, two fills meeting exactly at the border would touch
        # and get merged right back together during 3D connected-component
        # labeling. The original detected pixels are unioned back in
        # unconditionally so this erosion can only ever trim *extra* fill
        # near a neighbor - it can never eat into real, already-detected
        # signal and reintroduce a hollow bite out of the object's own wall.
        owned = ndimage.binary_erosion(nearest_label == k, iterations=1, border_value=1)
        result |= (owned & filled_component) | component_mask

    return result.astype(np.uint8)


def _prepare_slice(slc, fill_solid, close_kernel):
    """Threshold + (optionally) solidify a single mask slice. Split out as a
    top-level function so it can be dispatched across threads."""
    binary = to_binary_mask(slc)
    if binary is None:
        return None
    if fill_solid:
        binary = solidify_mask_slice(binary, close_kernel=close_kernel)
    return binary.astype(np.uint8)


def build_solid_volume(mask_slices, fill_solid=True, close_kernel=_DEFAULT_GAP_REPAIR_KERNEL):
    """Convert a stack of 2D masks into a fully solid 3D volume, always at full
    resolution. Downsampling (the "step" the user picks in the UI) is applied
    later, per already-identified object, in generate_3d_mesh - never here.

    This ordering matters: if you downsample the whole volume first and THEN
    ask which voxels are connected to which, a block-max-pooling step can
    swallow the last 1-2px gap between two real, distinct, closely-packed
    axons and silently fuse them into one connected component - exactly the
    "everything is one blob / one color" failure. Labeling at full resolution
    first, then downsampling each object's own crop in isolation, makes that
    class of bug structurally impossible: two objects that were correctly
    identified as separate can never be re-merged by a resolution change
    afterwards.
    """
    # Each slice is independent, and the underlying cv2 calls (morphologyEx,
    # floodFill, findContours) release the GIL, so a thread pool gives a real
    # speedup here instead of processing slices one at a time.
    max_workers = min(32, (os.cpu_count() or 4) * 2)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(
            _prepare_slice,
            mask_slices,
            (fill_solid for _ in mask_slices),
            (close_kernel for _ in mask_slices),
        )
        prepared = [r for r in results if r is not None]

    if not prepared:
        return None
    volume = np.stack(prepared, axis=0).astype(np.uint8)

    if fill_solid:
        # Only close along Z (fixes slice-to-slice flicker/misalignment of the
        # SAME object). A full 3x3x3 box here would also bridge two distinct,
        # merely-adjacent-in-XY axons across neighboring Z slices - exactly the
        # kind of unwanted merge we're trying to avoid, since object separation
        # in XY is already handled correctly (and deliberately) above.
        z_structure = np.zeros((3, 3, 3), dtype=bool)
        z_structure[:, 1, 1] = True
        volume_bool = ndimage.binary_closing(volume.astype(bool), structure=z_structure)
        # Cap the Z-ends with an empty slice before filling. binary_fill_holes only
        # fills cavities fully enclosed on all sides - if a tube's lumen touches the
        # very first/last slice of the stack (e.g. the axon continues beyond the
        # imaged crop), that cavity technically "touches the border" and would
        # otherwise be skipped. Capping guarantees any lumen within the imaged
        # volume still gets filled, on top of the per-slice fill above.
        capped = np.pad(volume_bool, ((1, 1), (0, 0), (0, 0)), mode='constant', constant_values=False)
        capped = ndimage.binary_fill_holes(capped)
        volume = capped[1:-1].astype(np.uint8)

    return volume.astype(np.uint8)


def _downsample_block_max(vol, step):
    """Block-max-pool downsample by `step` in all 3 axes. Pads with background
    (zero) up to the next multiple of `step` instead of truncating, so no real
    foreground at the edge of a crop is ever silently dropped."""
    step = max(1, int(step))
    if step == 1:
        return vol
    z, y, x = vol.shape
    pad_z, pad_y, pad_x = (-z) % step, (-y) % step, (-x) % step
    if pad_z or pad_y or pad_x:
        vol = np.pad(vol, ((0, pad_z), (0, pad_y), (0, pad_x)), mode='constant', constant_values=0)
    z2, y2, x2 = vol.shape
    vol = vol.reshape(z2 // step, step, y2 // step, step, x2 // step, step)
    return vol.max(axis=(1, 3, 5))


# ==========================================
# 1. GLOBAL CONFIGURATION & MODEL INITIALIZATION
# ==========================================
if torch.cuda.is_available():
    device = torch.device("cuda") 
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")  
else:
    device = torch.device("cpu")
print(f"Hardware Engine Engaged: {device}")

WEIGHTS_PATH = "structure_model.pth"
SAM_CHECKPOINT = "sam_vit_b_01ec64.pth"

def initialize_unet():
    model = smp.Unet(encoder_name="resnet18", encoder_weights=None, in_channels=1, classes=1)
    if os.path.exists(WEIGHTS_PATH):
        try:
            state_dict = torch.load(WEIGHTS_PATH, map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
        except Exception as exc:
            print(f"Warning: could not load {WEIGHTS_PATH}: {exc}")
    model.to(device)
    return model

def initialize_sam():
    if not os.path.exists(SAM_CHECKPOINT):
        print("Downloading Meta SAM weights (~375MB)...")
        urllib.request.urlretrieve("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth", SAM_CHECKPOINT)
    print("Loading SAM into VRAM...")
    sam = sam_model_registry["vit_b"](checkpoint=SAM_CHECKPOINT)
    sam.to(device=device)
    return SamPredictor(sam)

unet_model = initialize_unet()
sam_predictor = initialize_sam()

# ==========================================
# 2. CORE PREPROCESSING (ADVANCED PIPELINE)
# ==========================================
def load_and_preprocess_em(image_path, clip_limit=3.0, tile_size=8):
    image_path = resolve_file_path(image_path)
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY) if img.shape[2] == 4 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    if gray.dtype == np.uint16:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    elif gray.dtype != np.uint8:
        max_val = float(gray.max()) if gray.size else 0.0
        gray = (gray / max_val * 255).astype(np.uint8) if max_val > 0 else np.zeros_like(gray, dtype=np.uint8)

    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

    # Robust percentile-based contrast normalization, instead of raw mean/std
    # z-scoring. Per-image mean/std is fragile: any single EM crop that happens
    # to have lower natural contrast (out-of-focus region, flatter patch, etc.)
    # has a small std, and dividing by a small std stretches that slice's noise
    # across the full 0-255 range. CLAHE's local contrast boost then amplifies
    # that stretched noise further, and the final sharpening pass exaggerates
    # it again - three compounding amplification steps, which is what produced
    # the "extremely grainy" output. Percentile clipping anchors the stretch to
    # the image's actual observed signal range instead of an easily-skewed
    # statistic, so a naturally flatter/noisier slice doesn't get blown out.
    p_low, p_high = np.percentile(denoised, (0.5, 99.5))
    if p_high <= p_low:
        p_low, p_high = float(denoised.min()), float(denoised.max())
    if p_high <= p_low:
        normalized = denoised.astype(np.uint8)
    else:
        normalized = np.clip(
            (denoised.astype(np.float64) - p_low) / (p_high - p_low) * 255, 0, 255
        ).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(tile_size), int(tile_size)))
    enhanced = clahe.apply(normalized)

    # CLAHE boosts LOCAL contrast, which includes boosting noise in otherwise
    # flat regions - a well-known side effect. A mild edge-preserving
    # (bilateral) smoothing pass here cleans up that noise while leaving real
    # structural edges intact, so the sharpening step below sharpens genuine
    # structure instead of re-emphasizing noise CLAHE just introduced.
    enhanced = cv2.bilateralFilter(enhanced, d=5, sigmaColor=25, sigmaSpace=25)

    inverted = 255 - enhanced

    kernel_sharpen = np.array([[-0.5, -0.5, -0.5],
                               [-0.5,  5.0, -0.5],
                               [-0.5, -0.5, -0.5]])

    sharpened = cv2.filter2D(inverted.astype(np.float32), -1, kernel_sharpen)
    final_output = np.clip(sharpened, 0, 255).astype(np.uint8)

    return final_output

# ==========================================
# 3. TAB 1: INFERENCE WITH COLOR OVERLAY
# ==========================================
def process_and_predict(image_path, clip_limit, tile_size, hex_color):
    if image_path is None: return None, None, "Please upload an image file."
    preprocessed_img = load_and_preprocess_em(image_path, clip_limit, tile_size)
    
    eval_img = cv2.resize(preprocessed_img, (512, 512))
    input_tensor = torch.from_numpy(eval_img.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)

    unet_model.eval()
    with torch.inference_mode(): 
        prediction = torch.sigmoid(unet_model(input_tensor)).squeeze().cpu().numpy()

    prediction = cv2.resize(prediction, (preprocessed_img.shape[1], preprocessed_img.shape[0]))
    binary_mask_bool = prediction > 0.5

    hex_color = hex_color.lstrip('#')
    rgb_color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    img_rgb = cv2.cvtColor(preprocessed_img, cv2.COLOR_GRAY2RGB)
    color_layer = img_rgb.copy()
    color_layer[binary_mask_bool] = rgb_color
    
    overlay_result = cv2.addWeighted(color_layer, 0.4, img_rgb, 0.6, 0)

    return preprocessed_img, overlay_result, "Inference successful."

# ==========================================
# 4. TAB 2: ADVANCED SAM ENGINE WITH UNDO & SCALES
# ==========================================
def create_sam_plot(overlay):
    fig = go.Figure()
    fig.add_trace(go.Image(z=overlay))
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        dragmode='pan',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def load_workspace(image_path, clip_limit, tile_size):
    image_path = resolve_file_path(image_path)
    if image_path is None: return None, None, None, [], []
    clahe_img = load_and_preprocess_em(image_path, clip_limit, tile_size)

    img_rgb = cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2RGB)
    sam_predictor.set_image(img_rgb)
    empty_mask = np.zeros_like(clahe_img)
    
    return img_rgb, empty_mask, create_sam_plot(img_rgb), [], []

def run_sam_inference(rgb_state, global_mask, points, labels, scale_idx):
    if not points:
        overlay = rgb_state.copy()
        overlay[global_mask > 0] = [0, 150, 255]
        return np.zeros(global_mask.shape, dtype=np.uint8), overlay
        
    masks, scores, logits = sam_predictor.predict(
        point_coords=np.array(points),
        point_labels=np.array(labels),
        multimask_output=True,
    )
    
    selected_mask = (masks[scale_idx] * 255).astype(np.uint8)
    
    overlay = rgb_state.copy()
    overlay[global_mask > 0] = [0, 150, 255]        
    overlay[selected_mask > 0] = [0, 255, 0]        
    
    for pt, lbl in zip(points, labels):
        color = (0, 255, 0) if lbl == 1 else (255, 0, 0)
        cv2.circle(overlay, (pt[0], pt[1]), 4, color, -1)
        
    return selected_mask, overlay

def sam_click_plotly(click_mode, mask_scale, rgb_state, global_mask, current_points, current_labels, x, y):
    if rgb_state is None: return None, current_points, current_labels, None
    label = 1 if "Positive" in click_mode else 0
    new_points = current_points + [[int(x), int(y)]]
    new_labels = current_labels + [label]
    scale_idx = int(mask_scale) - 1
    
    preview_mask, overlay = run_sam_inference(rgb_state, global_mask, new_points, new_labels, scale_idx)
    return preview_mask, new_points, new_labels, create_sam_plot(overlay)

def undo_last_click(mask_scale, rgb_state, global_mask, current_points, current_labels):
    if not current_points: 
        overlay = rgb_state.copy()
        if global_mask is not None: overlay[global_mask > 0] = [0, 150, 255]
        return None, current_points, current_labels, create_sam_plot(overlay)
        
    new_points = current_points[:-1]
    new_labels = current_labels[:-1]
    scale_idx = int(mask_scale) - 1
    
    preview_mask, overlay = run_sam_inference(rgb_state, global_mask, new_points, new_labels, scale_idx)
    return preview_mask, new_points, new_labels, create_sam_plot(overlay)

def update_scale(mask_scale, rgb_state, global_mask, current_points, current_labels):
    scale_idx = int(mask_scale) - 1
    preview_mask, overlay = run_sam_inference(rgb_state, global_mask, current_points, current_labels, scale_idx)
    return preview_mask, create_sam_plot(overlay)

def commit_selection(global_mask, preview_mask, rgb_state):
    if preview_mask is None: return global_mask, [], [], create_sam_plot(rgb_state)
    updated_global = cv2.bitwise_or(global_mask, preview_mask)
    overlay = rgb_state.copy()
    overlay[updated_global > 0] = [0, 150, 255]
    return updated_global, [], [], create_sam_plot(overlay)

def clear_selection(global_mask, rgb_state):
    overlay = rgb_state.copy()
    overlay[global_mask > 0] = [0, 150, 255]
    return global_mask, [], [], create_sam_plot(overlay)

def push_to_manual(mask_state):
    return mask_state if mask_state is not None else None

def save_ground_truth(image_path, editor_data, mask_state, dataset_dir):
    image_path = resolve_file_path(image_path)
    if image_path is None or not dataset_dir: return "Error: Missing data."

    final_mask = None
    if isinstance(editor_data, dict) and "composite" in editor_data:
        final_mask = editor_data["composite"]
        if len(final_mask.shape) == 3: final_mask = cv2.cvtColor(final_mask, cv2.COLOR_BGR2GRAY)
    elif mask_state is not None:
        final_mask = mask_state

    if final_mask is None: return "Error: No mask."

    _, final_mask = cv2.threshold(final_mask, 127, 255, cv2.THRESH_BINARY)
    os.makedirs(os.path.join(dataset_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, "masks"), exist_ok=True)

    filename = f"sample_{int(time.time())}"
    ext = os.path.splitext(image_path)[1] or ".png"

    shutil.copy(image_path, os.path.join(dataset_dir, "images", f"{filename}{ext}"))
    cv2.imwrite(os.path.join(dataset_dir, "masks", f"{filename}.png"), final_mask)
    return f"Success! Saved {filename}."

# ==========================================
# 5. TAB 3: OPTIMIZED LOCAL TRAINING
# ==========================================
class SegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, clip_limit, tile_size):
        self.img_dir, self.mask_dir, self.clip_limit, self.tile_size = img_dir, mask_dir, clip_limit, tile_size
        self.filenames = [f for f in os.listdir(img_dir) if f.lower().endswith((".tiff", ".tif", ".png", ".jpg", ".jpeg"))]
        # load_and_preprocess_em (denoise + CLAHE + sharpen) is the expensive part
        # of __getitem__ and clip_limit/tile_size are fixed for the whole training
        # run, so cache its output per-file instead of recomputing it every epoch.
        self._cache = {}

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        name = self.filenames[idx]
        if name in self._cache:
            img, mask = self._cache[name]
            return torch.from_numpy(img).unsqueeze(0), torch.from_numpy(mask).unsqueeze(0)

        img = load_and_preprocess_em(os.path.join(self.img_dir, name), self.clip_limit, self.tile_size)
        mask_path = os.path.join(self.mask_dir, f"{os.path.splitext(name)[0]}.png")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None: raise FileNotFoundError(f"Missing mask for image {name}: {mask_path}")

        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)

        img = img.astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.float32)
        self._cache[name] = (img, mask)
        return torch.from_numpy(img).unsqueeze(0), torch.from_numpy(mask).unsqueeze(0)

def run_local_training(img_dir, mask_dir, epochs, lr, batch_size, clip_limit, tile_size, progress=gr.Progress()):
    global unet_model

    if not img_dir or not os.path.isdir(img_dir): return "Error: Raw images folder does not exist."
    if not mask_dir or not os.path.isdir(mask_dir): return "Error: Masks folder does not exist."

    dataset = SegmentationDataset(img_dir, mask_dir, clip_limit, tile_size)
    if len(dataset) == 0: return "Error: No training images found."

    cpu_workers = min(4, os.cpu_count() or 1)
    dataloader = DataLoader(
        dataset, batch_size=int(batch_size), shuffle=True,
        num_workers=cpu_workers, pin_memory=(device.type == 'cuda'),
        # Keep worker processes (and each one's per-image preprocessing cache)
        # alive across epochs instead of respawning them every epoch, which
        # would otherwise wipe the cache above and re-pay the preprocessing
        # cost every single epoch.
        persistent_workers=(cpu_workers > 0),
    )

    unet_model.train()
    optimizer = torch.optim.Adam(unet_model.parameters(), lr=float(lr))
    dice_loss = smp.losses.DiceLoss(mode="binary", from_logits=True)
    bce_loss = torch.nn.BCEWithLogitsLoss()

    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler(device.type) if use_amp else None

    total_steps = max(1, int(epochs) * len(dataloader))
    step = 0

    for epoch in range(int(epochs)):
        for imgs, masks in dataloader:
            imgs, masks = imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = unet_model(imgs)
                loss = dice_loss(logits, masks) + bce_loss(logits, masks)

            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            step += 1
            progress(step / total_steps, desc=f"Epoch {epoch + 1}/{int(epochs)} - Loss: {loss.item():.4f}")

    torch.save(unet_model.state_dict(), WEIGHTS_PATH)
    # The weights already in memory are identical to what was just saved, so
    # just flip to eval mode instead of paying for a redundant disk round-trip
    # through initialize_unet() (re-building the architecture + re-reading the
    # state dict we ourselves just wrote).
    unet_model.eval()

    if device.type == 'cuda': torch.cuda.empty_cache()
    elif device.type == 'mps': torch.mps.empty_cache()

    return f"Training completed. Weights saved to {WEIGHTS_PATH}."

# ==========================================
# 6. TAB 4: HIGH-PERFORMANCE 3D ENGINE
# ==========================================
import colorsys

def _object_color(index, saturation=0.68, value=0.95):
    """Golden-angle hue rotation: each successive object's hue is placed as far
    as possible from every previous one, so any number of objects - 9 or 90 -
    stay visually distinct instead of cycling through a short fixed palette
    and producing color collisions between unrelated, nearby axons."""
    hue = (index * 0.6180339887498949) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


def _shadow_footprint(x, y, floor_z):
    """Convex-hull footprint of an object flattened onto a floor plane, used to
    render a soft contact shadow beneath it for extra depth cues."""
    pts = np.column_stack([x, y]).astype(np.float32)
    if len(pts) < 3:
        return None
    hull = cv2.convexHull(pts).reshape(-1, 2)
    n = len(hull)
    if n < 3:
        return None
    hz = np.full(n, floor_z, dtype=np.float32)
    ii = np.zeros(n - 2, dtype=int)
    jj = np.arange(1, n - 1)
    kk = np.arange(2, n)
    return hull[:, 0], hull[:, 1], hz, ii, jj, kk


def generate_3d_mesh(mask_dir, step, z_ratio, opacity_val, fill_solid, gap_kernel=_DEFAULT_GAP_REPAIR_KERNEL):
    mask_dir = resolve_file_path(mask_dir)
    if not mask_dir or not os.path.exists(mask_dir):
        return None, "Error: Folder does not exist.", gr.update(choices=[], value=None)

    files = sorted([f for f in os.listdir(mask_dir) if f.lower().endswith(('.png', '.tif', '.tiff', '.jpg', '.jpeg'))])
    if len(files) < 3: return None, "Error: Need at least 3 mask slices.", gr.update(choices=[], value=None)

    # Disk reads are I/O bound - load slices concurrently instead of one at a time.
    file_paths = [os.path.join(mask_dir, f) for f in files]
    io_workers = min(32, (os.cpu_count() or 4) * 4)
    with concurrent.futures.ThreadPoolExecutor(max_workers=io_workers) as executor:
        loaded = list(executor.map(lambda p: cv2.imread(p, cv2.IMREAD_GRAYSCALE), file_paths))
    mask_slices = [img for img in loaded if img is not None]

    step = max(1, int(step))
    raw_fg_pixels = int(sum(np.count_nonzero(to_binary_mask(s)) for s in mask_slices))

    # Always built at FULL resolution - downsampling happens later, per object.
    vol_arr = build_solid_volume(mask_slices, fill_solid=fill_solid, close_kernel=gap_kernel)

    if vol_arr is None or vol_arr.size == 0 or vol_arr.sum() == 0:
        return None, "Error: Masks are completely blank.", gr.update(choices=[], value=None)

    filled_fg_pixels = int(vol_arr.sum())
    fill_note = ""
    if fill_solid:
        approx_raw = max(1, raw_fg_pixels)
        if filled_fg_pixels <= approx_raw * 1.05:
            fill_note = (
                " | Note: solid-fill made little to no difference on this data "
                "- your masks were likely already solid."
            )
        else:
            growth = filled_fg_pixels / approx_raw
            fill_note = f" | Solid-fill expanded foreground {growth:.1f}x (holes closed)."
    else:
        fill_note = " | Fill Interior is OFF - rendering raw (possibly hollow) masks as-is."

    vol_arr = np.pad(vol_arr, pad_width=1, mode='constant', constant_values=0)
    # Labeling at full resolution, BEFORE any downsampling, is what guarantees
    # two genuinely-separate nearby objects can never later get silently fused
    # by a downsample step (see build_solid_volume's docstring).
    labeled_vol = measure.label(vol_arr.astype(bool), connectivity=3)

    if labeled_vol.max() == 0:
        return None, "Error: Could not render meshes.", gr.update(choices=[], value=None)

    fig = go.Figure()

    object_names = []
    mesh_traces = []
    box_traces = []

    slices = ndimage.find_objects(labeled_vol)
    z_spacing = float(z_ratio) * step
    xy_spacing = float(step)

    vol_shape = labeled_vol.shape

    def _extract_component_mesh(label_idx, slc):
        """Run marching_cubes for a single connected component. Independent
        per object, so this is dispatched across a thread pool below.

        Crops this object out of the FULL-RESOLUTION labeled volume, then
        downsamples ONLY this isolated crop by `step` before marching_cubes.
        Because every other object has already been zeroed out of this crop,
        no amount of downsampling here can merge it with a neighbor - that
        correctness was already locked in when `labeled_vol` was computed.

        ndimage.find_objects gives the TIGHTEST possible bounding box, meaning
        the object's own mask touches every edge of that crop by definition.
        marching_cubes has no "outside" to interpolate against there, so it
        can't draw a closing face at that boundary - the resulting mesh comes
        out open (missing end-caps) exactly at the object's tightest extent.
        This is invisible for a compact blob but very visible for a thin-walled
        tube/ring like a myelin sheath, where the missing cap reads as a hole
        in the wall. Expanding the crop by `step` voxels of real zero-background
        on every side (clamped to the volume bounds) - enough to survive the
        downsample below and still leave a real zero margin - gives
        marching_cubes something to close against, so every object comes out
        fully watertight.
        """
        margin = max(1, step)
        z0 = max(0, slc[0].start - margin); z1 = min(vol_shape[0], slc[0].stop + margin)
        y0 = max(0, slc[1].start - margin); y1 = min(vol_shape[1], slc[1].stop + margin)
        x0 = max(0, slc[2].start - margin); x1 = min(vol_shape[2], slc[2].stop + margin)

        cropped_vol = labeled_vol[z0:z1, y0:y1, x0:x1]
        component_mask = (cropped_vol == label_idx).astype(np.uint8)
        if component_mask.sum() < 50:
            return None

        component_mask = _downsample_block_max(component_mask, step)

        try:
            verts, faces, _, _ = measure.marching_cubes(
                component_mask,
                level=0.5,
                spacing=(z_spacing, xy_spacing, xy_spacing),
            )
        except Exception:
            return None

        # z0/y0/x0 are full-resolution voxel indices, so they're converted to
        # physical units using the full-res voxel size (z_ratio, 1.0) - NOT
        # z_spacing/xy_spacing, which are the size of one *downsampled* voxel.
        verts[:, 0] += z0 * float(z_ratio)
        verts[:, 1] += y0
        verts[:, 2] += x0
        return verts, faces

    # marching_cubes per object is the main CPU cost of this tab and each object
    # is fully independent, so extract them concurrently across a thread pool
    # (skimage's implementation releases the GIL for the core computation).
    cpu_workers = min(8, os.cpu_count() or 4)
    with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_workers) as executor:
        futures = [
            executor.submit(_extract_component_mesh, idx + 1, slc)
            for idx, slc in enumerate(slices) if slc is not None
        ]
        mesh_results = [f.result() for f in futures]

    # A shared floor for every object's contact shadow, so they all sit on the
    # same visual "ground" instead of each floating at its own local minimum.
    valid_results = [r for r in mesh_results if r is not None]
    global_floor_z = min((v[:, 0].min() for v, _ in valid_results), default=0.0)

    # Trace building stays sequential so object numbering / colors stay
    # deterministic across runs.
    count = 0
    for result in mesh_results:
        if result is None:
            continue
        verts, faces = result

        object_id = f"Object {count + 1}"
        object_color = _object_color(count)
        object_names.append(object_id)

        x, y, z = verts[:, 2], verts[:, 1], verts[:, 0]

        shadow = _shadow_footprint(x, y, global_floor_z)
        if shadow is not None:
            sx, sy, sz, si, sj, sk = shadow
            mesh_traces.append(go.Mesh3d(
                x=sx, y=sy, z=sz, i=si, j=sj, k=sk,
                color='rgb(10, 10, 15)', opacity=0.30,
                lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0),
                hoverinfo='skip', showlegend=False,
            ))

        mesh_traces.append(go.Mesh3d(
            x=x, y=y, z=z,
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            opacity=float(opacity_val), color=object_color, name=object_id,
            flatshading=False, hoverinfo='name',
            # Angled key light + soft ambient fill for a glossy, well-rounded
            # look: strong enough diffuse/specular to show a clear light/shadow
            # gradient across each tube's own curved surface, without blowing
            # out highlights.
            lighting=dict(ambient=0.32, diffuse=0.75, roughness=0.35, specular=0.55, fresnel=0.18),
            lightposition=dict(x=200, y=250, z=1200),
            # A thin dark contour along silhouette/ridge edges reads like subtle
            # ambient occlusion and helps separate overlapping/adjacent objects.
            contour=dict(show=True, color='#1a1a1a', width=1),
        ))

        min_x, max_x = x.min(), x.max()
        min_y, max_y = y.min(), y.max()
        min_z, max_z = z.min(), z.max()

        bx = [min_x, max_x, max_x, min_x, min_x, None, min_x, max_x, max_x, min_x, min_x, None, min_x, min_x, None, max_x, max_x, None, max_x, max_x, None, min_x, min_x]
        by = [min_y, min_y, max_y, max_y, min_y, None, min_y, min_y, max_y, max_y, min_y, None, min_y, min_y, None, min_y, min_y, None, max_y, max_y, None, max_y, max_y]
        bz = [min_z, min_z, min_z, min_z, min_z, None, max_z, max_z, max_z, max_z, max_z, None, min_z, max_z, None, min_z, max_z, None, min_z, max_z, None, min_z, max_z]

        box_traces.append(go.Scatter3d(
            x=bx, y=by, z=bz, mode='lines',
            line=dict(color='#FFFF00', width=8),
            name=f"Box {object_id}", hoverinfo='skip', visible=False, showlegend=False
        ))

        count += 1

    if count == 0: return None, "Error: Could not render meshes.", gr.update(choices=[], value=None)

    for t in mesh_traces: fig.add_trace(t)
    for t in box_traces: fig.add_trace(t)

    studio_axis = dict(
        showbackground=True, backgroundcolor='rgb(235, 238, 242)',
        gridcolor='rgb(210, 214, 220)', zerolinecolor='rgb(210, 214, 220)',
        showspikes=False,
    )
    fig.update_layout(
        scene=dict(
            aspectmode='data',
            xaxis=dict(title='X', **studio_axis),
            yaxis=dict(title='Y', **studio_axis),
            zaxis=dict(title='Depth (Z)', **studio_axis),
            camera=dict(eye=dict(x=1.5, y=-1.5, z=0.9)),
        ),
        paper_bgcolor='rgb(248, 249, 251)',
        margin=dict(l=0, r=0, b=0, t=0),
        legend=dict(title="Detected Structures")
    )

    return fig, f"Successfully rendered {count} individual structures.{fill_note}", gr.update(choices=object_names, value=None)

def update_plot_selection(fig_dict, selected_object, opacity_val):
    if not fig_dict: return gr.update()
    if hasattr(fig_dict, "to_dict"): fig_dict = fig_dict.to_dict()
    if 'data' not in fig_dict: return gr.update()

    base_opacity = float(opacity_val)
    # When something IS selected, dim everything else well below the base
    # opacity rather than leaving it at `base_opacity` - otherwise, at the
    # default Global Base Opacity of 1.0, a selected object (opacity 1.0) and
    # every unselected object (also opacity 1.0) look identical and the
    # highlight is invisible.
    unselected_opacity = min(base_opacity, 0.12) if selected_object else base_opacity

    for trace in fig_dict['data']:
        name = trace.get('name', '')
        if name.startswith('Object '):
            # Only ever touch opacity here - color is left completely alone.
            trace['opacity'] = 1.0 if selected_object and name == selected_object else unselected_opacity
        elif name.startswith('Box Object '):
            expected_box = f"Box {selected_object}"
            trace['visible'] = bool(selected_object and name == expected_box)

    return fig_dict

def deselect_axon(fig_dict, opacity_val):
    updated_fig = update_plot_selection(fig_dict, None, opacity_val)
    return updated_fig, gr.update(value=None)


# ==========================================
# 7. GRADIO DASHBOARD LAYOUT
# ==========================================
with gr.Blocks(title="Custom Segmentation Hub") as demo:
    gr.Markdown("# 🔬 Custom Local Segmentation Hub")

    with gr.Accordion("⚙️ Global Preprocessing Settings (CLAHE)", open=False):
        clip_slider = gr.Slider(minimum=1.0, maximum=10.0, value=3.0, step=0.5, label="Clip Limit")
        grid_slider = gr.Slider(minimum=4, maximum=32, value=8, step=4, label="Tile Grid Size")

    with gr.Tabs():
        # --- TAB 1 (UPDATED COLOR OVERLAY) ---
        with gr.TabItem("🎯 1. Segmentation (Inference)"):
            with gr.Row():
                with gr.Column():
                    inf_file = gr.File(label="Upload Raw EM File", file_types=[".tiff", ".tif", ".png", ".jpg"])
                    mask_color = gr.ColorPicker(label="Prediction Overlay Color", value="#0096FF")
                    segment_btn = gr.Button("Segment Structure", variant="primary")
                    inf_status = gr.Textbox(label="Status", interactive=False)
                with gr.Column():
                    clahe_output = gr.Image(label="CLAHE Enhanced Image", image_mode="L")
                    inf_mask_output = gr.Image(label="AI Prediction Overlay", image_mode="RGB")
            segment_btn.click(fn=process_and_predict, inputs=[inf_file, clip_slider, grid_slider, mask_color], outputs=[clahe_output, inf_mask_output, inf_status])

        # --- TAB 2 (ADVANCED SAM) ---
        with gr.TabItem("🖌️ 2. Smart SAM Annotation Studio"):
            rgb_state = gr.State()
            global_mask_state = gr.State()
            preview_mask_state = gr.State()
            points_state = gr.State([])
            labels_state = gr.State([])
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Step 1: Initialize")
                    anno_file = gr.File(label="Upload Raw EM File", file_types=[".tiff", ".tif", ".png", ".jpg"])
                    load_btn = gr.Button("🔄 Load Image", variant="secondary")
                    
                    gr.Markdown("---")
                    gr.Markdown("### Step 2: SAM Controls")
                    click_mode = gr.Radio(choices=["🟢 Positive Point (Add)", "🔴 Negative Point (Exclude)"], value="🟢 Positive Point (Add)", label="1. Brush Type")
                    mask_scale = gr.Slider(minimum=1, maximum=3, step=1, value=1, label="2. SAM Area Guess (Slide if mask is too big/small)")
                    
                    with gr.Row():
                        undo_btn = gr.Button("↩️ Undo Last Click")
                        clear_btn = gr.Button("❌ Clear Current Object")
                        
                    commit_btn = gr.Button("✔️ COMMIT OBJECT (Must click before moving to next structure!)", variant="primary")
                        
                    gr.Markdown("---")
                    gr.Markdown("### Step 3: Save to Dataset")
                    dataset_path = gr.Textbox(label="Main Dataset Folder Path", placeholder="/home/user/my_dataset")
                    save_btn = gr.Button("💾 Save Image & Mask", variant="primary")
                    anno_status = gr.Textbox(label="Save Status", interactive=False)

                    # HIDDEN UI HOOKS FOR NATIVE SAM SCROLL-ZOOM SUPPORT
                    sam_x = gr.Number(visible=False, elem_id="sam_x")
                    sam_y = gr.Number(visible=False, elem_id="sam_y")
                    sam_btn = gr.Button("Trigger SAM", visible=False, elem_id="sam_btn")

                with gr.Column():
                    clickable_sam_display = gr.Plot(label="SAM Interactive View", elem_id="sam_plot")
                    
                    gr.Markdown("### Optional: Manual Retouch")
                    push_btn = gr.Button("⬇️ Push Committed Masks to Canvas for Final Polish")
                    manual_editor = gr.ImageEditor(label="Manual Canvas", type="numpy", image_mode="L", brush=gr.Brush(colors=["#FFFFFF", "#000000"], color_mode="fixed"))

            load_btn.click(
                fn=load_workspace, inputs=[anno_file, clip_slider, grid_slider], 
                outputs=[rgb_state, global_mask_state, clickable_sam_display, points_state, labels_state]
            )
            
            sam_btn.click(
                fn=sam_click_plotly,
                inputs=[click_mode, mask_scale, rgb_state, global_mask_state, points_state, labels_state, sam_x, sam_y],
                outputs=[preview_mask_state, points_state, labels_state, clickable_sam_display]
            )

            clickable_sam_display.change(
                fn=None,
                js="""
                function() {
                    setTimeout(() => {
                        let plots = document.querySelectorAll('#sam_plot .js-plotly-plot');
                        plots.forEach((plot) => {
                            if (!plot._has_zoom_patched) {
                                Plotly.setPlotConfig({scrollZoom: true});
                                Plotly.relayout(plot, {'xaxis.fixedrange': false, 'yaxis.fixedrange': false});
                                plot._has_zoom_patched = true;
                            }
                            if (plot.__sam_click_bound) return;
                            plot.__sam_click_bound = true;
                            plot.on('plotly_click', (data) => {
                                if (!data || !data.points) return;
                                let x = Math.round(data.points[0].x);
                                let y = Math.round(data.points[0].y);
                                let xInput = document.querySelector('#sam_x input');
                                let yInput = document.querySelector('#sam_y input');
                                let btn = document.querySelector('#sam_btn');
                                if (xInput && yInput && btn) {
                                    xInput.value = x;
                                    xInput.dispatchEvent(new Event('input', { bubbles: true }));
                                    yInput.value = y;
                                    yInput.dispatchEvent(new Event('input', { bubbles: true }));
                                    setTimeout(() => { btn.click(); }, 100);
                                }
                            });
                        });
                    }, 500);
                }
                """
            )

            mask_scale.change(
                fn=update_scale, inputs=[mask_scale, rgb_state, global_mask_state, points_state, labels_state],
                outputs=[preview_mask_state, clickable_sam_display]
            )
            undo_btn.click(
                fn=undo_last_click, inputs=[mask_scale, rgb_state, global_mask_state, points_state, labels_state],
                outputs=[preview_mask_state, points_state, labels_state, clickable_sam_display]
            )
            commit_btn.click(
                fn=commit_selection, inputs=[global_mask_state, preview_mask_state, rgb_state], 
                outputs=[global_mask_state, points_state, labels_state, clickable_sam_display]
            )
            clear_btn.click(fn=clear_selection, inputs=[global_mask_state, rgb_state], outputs=[global_mask_state, points_state, labels_state, clickable_sam_display])
            
            push_btn.click(fn=push_to_manual, inputs=[global_mask_state], outputs=[manual_editor])
            save_btn.click(fn=save_ground_truth, inputs=[anno_file, manual_editor, global_mask_state, dataset_path], outputs=[anno_status])

        # --- TAB 3 (OPTIMIZED) ---
        with gr.TabItem("🎛️ 3. Local Network Training"):
            with gr.Row():
                with gr.Column():
                    img_path_input = gr.Textbox(label="Raw Images Folder", placeholder="/home/user/my_dataset/images")
                    mask_path_input = gr.Textbox(label="Masks Folder", placeholder="/home/user/my_dataset/masks")
                    epoch_input = gr.Slider(minimum=1, maximum=200, value=20, step=1, label="Epochs")
                    batch_input = gr.Slider(minimum=1, maximum=16, value=4, step=1, label="Batch Size")
                    lr_input = gr.Dropdown(choices=["0.01", "0.001", "0.0001"], value="0.001", label="Learning Rate")
                    train_btn = gr.Button("🚀 Begin Local Training", variant="secondary")
                with gr.Column():
                    console_output = gr.Textbox(label="Training Execution Summary", interactive=False, lines=6)
            train_btn.click(fn=run_local_training, inputs=[img_path_input, mask_path_input, epoch_input, lr_input, batch_input, clip_slider, grid_slider], outputs=console_output)

        # --- TAB 4 (ULTRA-FAST RECONSTRUCTION) ---
        with gr.TabItem("🧊 4. 3D Volume Reconstruction"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 1. Build Volumetric Data")
                    mask_folder_3d = gr.Textbox(label="Masks Folder Path (Z-Stack)", placeholder="/path/to/predicted/masks")
                    downsample_slider = gr.Slider(minimum=1, maximum=8, value=2, step=1, label="Downsample Factor (Mesh Detail vs. Render Speed)")
                    z_ratio_slider = gr.Slider(minimum=0.1, maximum=5.0, value=1.0, step=0.1, label="Z-Axis Stretch (To fix flat EM slices)")
                    fill_solid_checkbox = gr.Checkbox(label="Fill Interior (Render as a solid structure)", value=True)
                    gap_kernel_slider = gr.Slider(
                        minimum=3, maximum=41, value=15, step=2,
                        label="Wall Gap-Repair Strength (raise if rings still look hollow)",
                        info="Starting size of the closing used to seal breaks in a ring's own outline. "
                             "Automatically grows further, per object, if a given object still isn't closing at this size."
                    )
                    mesh_opacity = gr.Slider(minimum=0.1, maximum=1.0, value=1.0, step=0.1, label="Global Base Opacity")
                    
                    render_btn = gr.Button("Calculate & Render 3D Volume", variant="primary")
                    render_status = gr.Textbox(label="Status", interactive=False)
                    
                    gr.Markdown("---")
                    gr.Markdown("### 2. High-Performance Object Selection")
                    gr.Markdown("Select an object from the dropdown menu to highlight it and draw a 3D bounding box around it.")

                    axon_dropdown = gr.Dropdown(label="Select Object", choices=[], interactive=True)
                    deselect_btn = gr.Button("❌ Deselect Object", variant="secondary")

                with gr.Column():
                    plot_output = gr.Plot(label="Interactive 3D Structure Viewer")
            
            # WIRING: Render Base Plot
            render_btn.click(
                fn=generate_3d_mesh, 
                inputs=[mask_folder_3d, downsample_slider, z_ratio_slider, mesh_opacity, fill_solid_checkbox, gap_kernel_slider], 
                outputs=[plot_output, render_status, axon_dropdown]
            )

            # WIRING: Instant Dictionary Update (Feeds Plot Output back into itself)
            axon_dropdown.change(
                fn=update_plot_selection,
                inputs=[plot_output, axon_dropdown, mesh_opacity],
                outputs=[plot_output]
            )
            
            # WIRING: Deselect Object
            deselect_btn.click(
                fn=deselect_axon,
                inputs=[plot_output, mesh_opacity],
                outputs=[plot_output, axon_dropdown]
            )

if __name__ == "__main__":
    demo.launch(inbrowser=True)