# ECE551-Video-Seg

Accurate segmentation masks are essential for training modern computer-vision models, yet creating them by hand remains time-consuming and error-prone.
ECE551-Video-Seg is a browser-based annotation application that aims to reduce this bottleneck by combining a conventional point-and-click user interface with **Meta’s Segment-Anything Model v2.1 (SAM 2.1)** running on a RESTful server.

The core workflow is as follows:

1. **User input** – The annotator selects a label and clicks once (or draws a box) on the object of interest.
2. **Model inference** – The backend feeds the image and prompt into SAM 2.1, which returns an initial mask in a few hundred milliseconds on a GPU.
3. **Interactive refinement** – The annotator can add positive or negative points to correct errors; each new prompt instantly updates the mask.
4. **Post-processing and export** – Masks are stored per image and exported in **COCO** format (JSON + RLE) for direct use in training pipelines.

Compared with traditional polygon tools, the system offers three advantages:

| Aspect | Polygon/Scribble tools | **ECE551-Video-Seg** |
|--------|-----------------------|----------------------|
| Time per image | Minutes (complex objects) | Tens of seconds (single prompt, few refinements) |
| Consistency | Variable between annotators | Deterministic model output, fewer systematic errors |
| Learning curve | Requires manual precision | Point-and-click; mask refinement is incremental |

The application is delivered as two Docker containers—one serving a Svelte front-end, the other a FastAPI back-end with PyTorch. GPU acceleration is optional but strongly recommended for practical throughput.

## 1 Main Features

| Category | Description |
|----------|-------------|
| Media management | Upload, list and preview images and videos. Media is stored per project. |
| Label management | Add, edit and delete labels. Each label is associated with a color. |
| Annotation studio | Prompt-based mask generation using SAM 2.1. Supports multiple labels, multiple prompts, undo/redo and keyboard shortcuts. Video support includes timeline navigation and object tracking. |
| Export | One-click export: COCO JSON for images, YouTube-VIS format for videos. |

## 2 Screenshots

### 2.1 Project Selector / Creation

![Projects](resources/projects.png)
![alt text](resources/new.png)

### 2.2 Project Dashboard

![Project](resources/project.png)

### 2.3 Media Gallery

![Gallery](resources/gallery.png)

### 2.4 Label Manager

![Labels](resources/labels.png)

### 2.5 Annotation Studio

![Studio](resources/studio.png)

## 3  System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA GPU, 8GB+ VRAM, CUDA 12+ (Required) | NVIDIA GPU, 16GB VRAM (for Large models) |
| CPU | 8 cores | 16 cores |
| RAM | 16 GB | 32 GB |
| Disk | 30 GB free | 50 GB free |
| OS | Linux, Windows (WSL2) | Linux (Ubuntu 22.04+) |
| Docker | 28.0+ with NVIDIA Container Toolkit | - |

## 4  Installation with Docker Compose

### 4.1 Prerequisites
1. **NVIDIA GPU** with updated drivers.
2. **Docker Desktop** or **Docker Engine** (Linux).
3. **NVIDIA Container Toolkit** must be installed and configured for Docker.
   - [Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

### 4.2 Start Application
```bash
git clone --recursive https://github.com/briyoon/ECE551-Video-Seg.git
cd ECE551-Video-Seg

# Build and start services (GPU support required)
docker compose up --build -d
```

Access the application at: http://localhost:3000

## 5 Manual Installation (Development)

For local development or environments without Docker.

### 5.1 Backend
Requires **Python 3.10+** and **CUDA 12+**. We recommend using `uv` for dependency management, but `pip` also works.

```bash
cd video_seg_backend

# Option A: Using uv (Recommended)
pip install uv
uv sync
source .venv/bin/activate

# Option B: Standard pip
pip install -e .
```

Start the API server (checkpoints will download automatically on first run):
```bash
python app.py
```
*Server runs on http://localhost:8000*

### 5.2 Frontend
Requires **Node.js 18+**.

```bash
cd video_seg_frontend
npm install
npm run dev
```
*Client runs on http://localhost:5173 (proxies API requests to port 8000)*


## 6 Detailed Workflow

The following section walks through every user-visible step in the workflow, and pairs each UI action with the relevant client-side component and back-end API call.

### 6.1 Project Home (“Projects”)

| Action | UI element (component) | Back-end call | Notes |
|--------|-----------------------|---------------|-------|
| Display list | `<Card>` tiles rendered by **Projects.svelte** (snippet 1) | `GET /api/v1/projects` *(handled in load function; not in snippet)* | Projects are passed to child components via Svelte context (`getContext('projectlist')`). |
| Create project | **“+ New Project”** button | — | Navigates to `/projects/new`. |

### 6.2 Create Project

| Action | UI element (component) | Back-end call | Validation / state |
|--------|-----------------------|---------------|---------------------|
| Enter name + select media type | `<Input>` & `<Select>` in **NewProject.svelte** (snippet 2) | — | Two-way bound to `name` and `media_type`. |
| Submit form | **Create Project** `<Button>` | `POST /api/v1/projects` → `createProjectApiV1ProjectsPost` | Disabled while `submitting=true`. |
| Success | Toast “Project created” (svelte-sonner) | — | `goto('/projects/{id}')` on success. |

### 6.3 Project Dashboard

Layout reference: `project.png`

| Tile | UI element | Destination | Purpose |
|------|------------|-------------|---------|
| **Gallery** | `<Button href="/gallery">` | Thumbnail view, upload / delete media. |
| **Labels** | `<Button href="/labels">` | Manage class list and colours. |
| **Studio** | `<Button href="/studio">` | Main annotation canvas. |
| **Export Annotations** | Secondary `<Button>` | Triggers `GET /api/v1/projects/{pid}/export` | Returns ZIP containing `coco.json` + mask files. |
| **Delete Project** | Destructive `<Button>` | `DELETE /api/v1/projects/{pid}` | Confirmation dialog then redirect to `/`. |

### 6.4 Label Manager

Layout reference: `labels.png`

| Action | Component (snippet 3) | API |
|--------|----------------------|-----|
| List labels | `listLabelsApiV1ProjectsPidLabelsGet` on mount (`refreshLabels()`). |
| Add label | “Add label” modal → `POST` | `createLabelApiV1ProjectsPidLabelsPost` |
| Edit label | Inline rename / recolour → `PATCH` | `updateLabelApiV1ProjectsPidLabelsLidPatch` |
| Delete label | Trash icon → confirm → `DELETE` | `deleteLabelApiV1ProjectsPidLabelsLidDelete` |

Internally each label is an object `{ id, name, color }`. Edits update the local `labels` array to keep the UI reactive.

### 6.5 Gallery / Media Manager

Layout reference: `gallery.png`

| Action | UI logic (snippet 4) | API |
|--------|---------------------|-----|
| Lazy grid render | `media.slice(0, visible)` with **IntersectionObserver** | — |
| Add media | “Add media” dialog → drag-and-drop or file picker | `POST /api/v1/projects/{pid}/media` |
| Preview full‐size | Click card → `<Dialog>` shows image/video | Static file route `/api/v1/projects/{pid}/media/{mid}` |
| Delete media | Trash icon in preview | `DELETE /api/v1/projects/{pid}/media/{mid}` |

*Accepted formats* are enforced on the client (`isValidFile`) and by the server:
- Images : `image/png`, `image/jpeg`
- Video  : `video/mp4`

### 6.6 Annotation Studio — Detailed Step-by-Step

This section expands the “Studio” step of the workflow and explains how each interaction is wired to the front-end component logic and the related API calls.

---

#### 6.6.1  Screen layout

| Region | DOM element(s) | Purpose |
|--------|----------------|---------|
| **Left sidebar** | `<aside>` (class `w-[270px]`) with two `Collapsible` lists | Navigate between *unannotated* and *annotated* media items. Each entry is a `<Button>` that sets `selected` via `selectMedia(item)`. |
| **Centre viewer** | `<main>` (class `relative flex-1`) | Displays the current image (or future video). Handles mouse events for adding / deleting prompts. Contains two overlay canvases: the static mask + prompt canvas (`annCanvas`) and the image/video element itself (`imgEl` or `<video>`). |
| **Right toolbar** | Second `<aside>` | Label selector, model selector, prompt list, undo / redo buttons, navigation shortcuts. |
| **Header** | `<ProjectNav>` breadcrumb | Shows project hierarchy and highlights active “Studio” route. |

---

#### 6.6.2  Data flow on initial load

1. **`onMount()`**
   ```ts
   refreshMedia();     // GET /media
   refreshModels();    // GET /models
   refreshLabels();    // GET /labels
   eventStream = getEventStream(); // open SSE /events
*Result*: sidebars are populated; the first media item becomes selected.

2. Media selection (selectMedia(item))
- Updates `selected`.
- Calls `refreshPrompts()` → `GET /prompts?media_id=…`.
- Calls `refreshAnnotations()` → `GET /annotations/image/{mid}`.
- Resets zoom / pan; redraws overlay.

#### 6.6.3 Adding a prompt (positive / negative)

| User gesture | Code path | HTTP call |
| --- | --- | --- |
| Left-click | `onclick` handler in `<main>` → `addPrompt(ix, iy, true)` | `POST /prompts/image` (body: click_label = 1) |
| Ctrl + Left-click | same → `addPrompt(ix, iy, false)` | `click_label = 0` |

addPrompt pushes a copy of the current prompt list to undoStack, clears redoStack, then sends the request.
If the backend returns { detail: "queued" }, the media-ID is put in the queued Set to show a spinner until the mask is computed. Server-side completion is signalled via Server-Sent Events on /events; the handler:

```ts
if (msg.event === 'annotations_updated') {
    queued.delete(msg.media_id);
    refreshPrompts();
    refreshAnnotations();
    refreshMedia();
}
```

#### 6.6.4 Deleting the nearest prompt

| Gesture |	Code | HTTP |
| --- | --- | --- |
| Right-click | `auxclick` handler → `deleteNearestPrompt(ix, iy)` | `DELETE /prompts/image/{aid}` |

`deleteNearestPrompt` finds the prompt with minimum Euclidean distance in pixel space and issues a `DELETE`. The result updates the same caches and overlay.

#### 6.6.5 Mask rendering pipeline

1. Mask retrieval – `refreshAnnotations()` calls
`GET /projects/{pid}/annotations/image/{mid}` which returns an array:

```json
[
  {
    "label_id": 2,
    "mask_rle": "27 3 32 5 …",
    …
  },
  …
]
```
2. RLE decoding (`decodeRle`) – converts the COCO “counts” string into a `Uint8Array` mask.

3. Compositing (`drawMasks`)
- Creates a scratch ImageData.
- Fills the alpha channel to 128 for interior pixels, 255 for dilated boundary (pseudo-outline).
- Uses label colour (labels[].color) for RGB.

4. Display – Scratch canvas is drawn onto `annCanvas`, then prompt dots are over-painted.

All transforms are applied via the 2-D canvas transform so the overlay stays aligned with the <img> element regardless of resize or device-pixel ratio.

#### 6.6.6 Keyboard shortcuts
| Key | Function | Implementation |
| --- | --- | ---
| 1 … 9 | Switch active label	| handleKey: numeric keys index into labels array. |
| ← / →	| Previous / next media |	prevImage() / nextImage() |
| U	| Next unannotated media |	nextUnannotated() |
Ctrl+Z	| Undo prompt change	| pops undoStack → pushes to redoStack |
Ctrl+Y	| Redo prompt change	| inverse of undo |

#### 6.6.7 Model selection
- Right toolbar `<Select>` lists available model IDs returned by GET /models.
- When the user changes selectedModelId, subsequent addPrompt() calls include it via ?model_key=….
- The server loads the corresponding SAM 2.1 checkpoint on first use and re-runs inference for that prompt.

#### 6.6.8 Undo / redo stacks
- Both stacks store deep copies (JSON.stringify/parse) of the prompts array only.
- Annotation pixels are not stored; masks are always requested fresh from the server. This prevents large memory use and guarantees consistency with server state.

#### 6.6.9 Failure handling
- Each API call is wrapped with a toast message on failure (toast.error).
- Network errors (non-2xx) are printed to console.error and surfaced to the user.

### 6.7 Video Annotation Workflow

Video annotation adds a temporal dimension using SAM 2's memory module.

1. **Load Video**: Clicking a video in the sidebar loads the frames and SAM 2 video state into GPU memory (`POST /video/load`).
2. **Timeline Navigation**: The scrubber allows navigating frames. Playback is supported with `requestAnimationFrame` for smooth rendering.
3. **Object Tracking**:
    - Add prompts on a single frame.
    - Click "submit" to propagate the mask to all video frames.
    - Uses `run_video_inference` with tracking to predict the object mask across time.
4. **Refinement**: Navigate to frames where tracking fails, add corrective prompts, and re-submit to update the object track.

### 6.8 Export
- Download JSON on the dashboard calls exportProject() (snippet 3).
- FastAPI streams a ZIP archive: coco.json + mask PNGs (or RLE only, based on settings).
- The browser creates a blob URL and triggers a download with the suggested filename project_{pid}.zip.

### 6.9 Deletion
Project-level deletion removes:
- Database records (project, labels, media, masks).
- On-disk media and mask files under DATA_DIR/{pid}.
Confirmation is handled on the client; irreversible action is performed by DELETE /api/v1/projects/{pid}.

## 7 Quick Evaluation

A small-scale study was conducted with the Oxford-IIIT Pet datasets. Both time and accuracy metrics were studied. A subset of 50 cats and 50 dogs were used, with even class distribution.

### Results:
Setup. 100 Oxford-IIIT Pet images (balanced cats/dogs). Ground truth derived from trimaps; predictions evaluated with COCO-style RLE using pycocotools.

Metrics used:

IoU (Intersection over Union):
Imagine two cut-out shapes: your prediction and the ground truth. IoU asks: “Out of everything covered by either shape, what fraction is the part where they overlap?” Bigger overlap → higher IoU.  Extra outside or missing areas lower it.

Dice:
Same two shapes, but Dice is a little more forgiving. It asks: “How big is the overlap compared to the sizes of the two shapes put together?” (and it counts the overlap twice). For the same masks, Dice is usually a bit higher than IoU.

Tiny example: If the overlap is “80% of everything either shape covers,” then IoU ≈ 0.80 and Dice ≈ 0.89.

Edge cases: Both empty → perfect match (often treated as 1.0). One empty, one not → 0.0.

#### Annotation Productivity

- Total time: 8:49 (mm:ss) → 529 s
- Throughput: 5.29 s/image (≈ 11.3 images/min, ~681 images/hour). This is an efficient rate for pixel-accurate masks and suggests an effective workflow and/or strong tool assistance.

#### Quality Metrics

- Mean IoU: 0.8008
- Mean Dice: 0.8871
- Class IoU: cat 0.8141, dog 0.7874 (∆ ≈ 0.0267)

#### Interpretation.

- IoU ≈ 0.80 / Dice ≈ 0.89 indicate high-quality masks with good overlap and boundary adherence.
- The modest cat–dog gap (~2.7 pts IoU) implies dogs are slightly harder—likely due to pose variability, fur edges, or occlusions.

#### Key Takeaways

- Speed and quality are both strong. You’re achieving sub-6-second masks with ~0.80 IoU on average.
- Small class disparity. Worth a targeted review of low-IoU dog cases to identify common failure modes.

### 7.1 Video Evaluation (DAVIS 2017)

To validate the system's tracking capabilities, we integrated a standardized evaluation pipeline using the **DAVIS 2017 (Densely Annotated Video Segmentation)** dataset.

#### 7.1.1 Methodology
The evaluation process consists of three automated stages, handled by `scripts/video_evaluation.py`:
1.  **Dataset Preparation**: A custom script (`scripts/create_davis_subset.py`) extracts a representative 10-video subset from the DAVIS validation split. It converts the original 480p frame sequences into H.264 MP4 files compatible with the web frontend and generates "Ground Truth Videos" where masks are burned into the footage for visual verification.
2.  **Inference & Export**: The user annotates the first frame (and optionally correction frames) in the UI, then exports the project in **YouTube-VIS** format.
3.  **Metric Calculation**: The evaluation script compares the exported JSON RLE masks against the ground-truth PNG masks frame-by-frame. We utilize the **J-Score (Region Jaccard)**, defined as the Intersection-over-Union (IoU) averaged across all frames for an object.

#### 7.1.2 Quantitative Results
We achieved a **Mean J-Score of 0.8571** on the evaluation subset using the `sam2.1_hiera_base_plus` model.

| category | J-Score | Analysis |
|----------|---------|----------|
| **Rigid Objects** | **> 0.95** | Objects with predictable motion (e.g., `car-turn`, `drift-chicane`) are tracked with near-perfect accuracy from a single prompt. |
| **Deformable** | **0.90 - 0.95**| SAM 2's memory module handles non-rigid deformation (e.g., `gold-fish`, `cat-girl`) robustly, maintaining boundary coherence. |
| **Occluded/Complex**| **0.53 - 0.75**| Scenes with heavy occlusion (e.g. `surf` - water covering the board) or multiple identical instances (`schoolgirls`) pose challenges. These represent the ideal use-case for the tool's "refinement" workflow, where adding 1-2 extra prompts in failing frames recovers the track. |

## 8 Technical Implementation: Video Support

Adding temporal consistency to a stateless web-tool required a fundamental re-architecture of both the backend inference engine and the frontend player.

### 8.1 Backend: Stateful Inference & Memory Banks
The core challenge of video segmentation is context. Unlike images, where predictions are independent, SAM 2 requires a "memory bank" of past frames to track objects.
- **Session Management**: We implemented a global `VIDEO_STATE_CACHE` (LRU) that persists the heavy GPU memory state (embeddings) between HTTP requests. This allows the stateless REST API to support stateful interactive editing.
- **Asynchronous Propagation**: Running inference on an entire video takes seconds to minutes. We enforce a non-blocking UI by offloading the `run_video_inference` loop to FastAPI `BackgroundTasks`. Progress is reported back to the client via a **Server-Sent Events (SSE)** channel (`/events`), allowing the user to see real-time updates without locking the browser.
- **Hybrid Rendering**: Transferring full-resolution 4K binary masks for every frame over the network is prohibitively slow. We implemented a **Server-Side Rendering (SSR)** pipeline:
    - **Edit Mode**: When paused, the frontend receives raw RLE masks to support precise clicking and editing.
    - **Playback**: During scrubbing/play, the backend composites masks into lightweight JPEG previews (`/frames/{i}/annotated`), guaranteeing 24fps performance even with complex segmentation.

### 8.2 Frontend: Frame-Accurate Synchronization
The standard HTML5 `<video>` element lacks the precision required for per-frame annotation (often having seeking inaccuracies of +/- 100ms).
- **Custom Player Implementation**: We replaced the native player with a `requestAnimationFrame`-driven engine that renders individual frame images.
- **Coordinate System**: To support responsive resizing while maintaining pixel-perfect prompts, we implemented a dual-coordinate transform system. It projects DOM event coordinates (client space) into "Video Source Space" using a computed letterbox offset (`ox`, `oy`) and scale factor. This ensures that a click on a pixel *visually* corresponds to the exact same pixel in the *tensor*, regardless of whether the browser window is 720p or 4K.
- **Optimistic State**: To mask network latency, the "Studio" UI implements an optimistic update queue (`pendingPrompts`). When a user adds a prompt, it is instantly rendered on the canvas locally via a "Yellow Dashed" indicator. The UI remains interactive while the backend processes the prompt and returns the confirmed mask, merging the states asynchronously.

### 8.3 Data & Export Standards
To Ensure compatibility with modern Video Object Segmentation (VOS) benchmarks, we adopted the **YouTube-VIS** annotation standard.
- **JSON Structure**: The dataset is exported as a single `coco.json`-like structure, but grouped by `tracks` instead of just images.
- **Segmentation Format**: Unlike standard COCO which often uses polygons, our export produces **RLE (Run-Length Encoding)** masks. This handles complex topologies (e.g., a donut shape or occluded object split in two) which polygons struggle to represent efficiently.

## 9 Future Goals

- Integration of an object-detection model to generate initial prompts automatically.
- Tools to further refine initial SAM2 annotations.
- Torch-TensorRT engine support for faster inference on compatible GPUs.
```