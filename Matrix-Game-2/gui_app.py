import os
import sys
import requests
import numpy as np

from PySide6 import QtCore, QtGui, QtWidgets


class GenerationWorker(QtCore.QObject):
    frameReady = QtCore.Signal(np.ndarray)
    status = QtCore.Signal(str)
    finished = QtCore.Signal()

    def __init__(self, config_path: str, checkpoint_path: str, pretrained_model_path: str, mode: str,
                 seed: int = 42, max_num_output_frames: int = 120, device: str = "auto",
                 actions_provider=None, image_path: str = ""):
        super().__init__()
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.pretrained_model_path = pretrained_model_path
        self.mode = mode
        self.seed = seed
        self.max_num_output_frames = max_num_output_frames
        self.device_str = device
        self.actions_provider = actions_provider
        self.image_path = image_path
        self._stop = False
        # Heavy ML components are set up inside run() to allow lightweight GUI-only env

    def stop(self):
        self._stop = True

    def _resizecrop(self, image, th, tw):
        w, h = image.size
        if h / w > th / tw:
            new_w = int(w)
            new_h = int(new_w * th / tw)
        else:
            new_h = int(h)
            new_w = int(new_h * tw / th)
        left = (w - new_w) / 2
        top = (h - new_h) / 2
        right = (w + new_w) / 2
        bottom = (h + new_h) / 2
        image = image.crop((left, top, right, bottom))
        return image

    def _on_frame(self, frame_rgb: np.ndarray):
        if self._stop:
            return
        self.frameReady.emit(frame_rgb)

    def _action_provider(self, mode: str):
        # actions_provider must return a dict compatible with cond_current replace arg: keys 'mouse' (tensor) and/or 'keyboard' (tensor)
        # Delegate to UI-provided callable if any, else default to no-op
        if self.actions_provider is not None:
            return self.actions_provider(mode)
        # Default neutral actions
        if mode == 'gta_drive':
            return {"mouse": [0.0, 0.0], "keyboard": [0, 0]}
        elif mode == 'templerun':
            return {"keyboard": [1, 0, 0, 0, 0, 0, 0]}
        else:
            return {"mouse": [0.0, 0.0], "keyboard": [0, 0, 0, 0]}

    @QtCore.Slot()
    def run(self):
        try:
            # Lazy imports so the GUI can run without ML deps installed
            import torch
            from omegaconf import OmegaConf
            from torchvision.transforms import v2
            from diffusers.utils import load_image
            from safetensors.torch import load_file
            from utils.misc import set_seed
            from utils.wan_wrapper import WanDiffusionWrapper
            from demo_utils.vae_block3 import VAEDecoderWrapper
            from pipeline import CausalInferenceStreamingPipeline
            from wan.vae.wanx_vae import get_wanx_vae_wrapper
            import traceback
            import sys
            import time

            def log(msg: str):
                ts = time.strftime('%H:%M:%S')
                self.status.emit(msg)
                print(f"[{ts}] [GUI] {msg}", flush=True)

            log("Selecting device…")
            # Auto-select device, fallback to CPU if CUDA not available
            dev_str = self.device_str
            if dev_str == "auto":
                dev_str = "cuda" if torch.cuda.is_available() else "cpu"
            elif dev_str == "cuda" and not torch.cuda.is_available():
                log("CUDA not available, falling back to CPU")
                dev_str = "cpu"
            device = torch.device(dev_str)
            # Use float16 on CUDA (safer with many kernels), float32 on CPU
            weight_dtype = torch.float16 if device.type == "cuda" else torch.float32

            frame_process = v2.Compose([
                v2.Resize(size=(352, 640), antialias=True),
                v2.ToTensor(),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])

            log(f"Seed={self.seed}; loading config: {self.config_path}")
            set_seed(self.seed)
            t0 = time.perf_counter()
            config = OmegaConf.load(self.config_path)
            log(f"Config loaded in {time.perf_counter()-t0:.2f}s")
            log("Initializing generator…")
            generator = WanDiffusionWrapper(**getattr(config, "model_kwargs", {}), is_causal=True)

            # VAE decoder
            log("Loading VAE decoder weights…")
            current_vae_decoder = VAEDecoderWrapper()
            vae_state_dict = torch.load(os.path.join(self.pretrained_model_path, "Wan2.1_VAE.pth"), map_location="cpu")
            decoder_state_dict = {}
            for key, value in vae_state_dict.items():
                if 'decoder.' in key or 'conv2' in key:
                    decoder_state_dict[key] = value
            current_vae_decoder.load_state_dict(decoder_state_dict)
            log("Moving VAE decoder to device…")
            current_vae_decoder.to(device, torch.float16 if device.type == "cuda" else torch.float32)
            current_vae_decoder.requires_grad_(False)
            current_vae_decoder.eval()
            log("Compiling VAE decoder…")
            current_vae_decoder.compile(mode="max-autotune-no-cudagraphs")

            log("Building pipeline…")
            pipeline = CausalInferenceStreamingPipeline(config, generator=generator, vae_decoder=current_vae_decoder)
            if self.checkpoint_path:
                log("Loading pretrained diffusion checkpoint…")
                state_dict = load_file(self.checkpoint_path)
                pipeline.generator.load_state_dict(state_dict)

            log("Moving pipeline to device…")
            pipeline = pipeline.to(device=device, dtype=weight_dtype)
            pipeline.vae_decoder.to(torch.float16 if device.type == "cuda" else torch.float32)

            # VAE encoder for image conditioning
            log("Initializing VAE encoder…")
            from wan.vae.wanx_vae import get_wanx_vae_wrapper
            vae = get_wanx_vae_wrapper(self.pretrained_model_path, torch.float16 if device.type == "cuda" else torch.float32)
            vae.requires_grad_(False)
            vae.eval()
            log("Moving VAE encoder to device…")
            vae = vae.to(device, weight_dtype)

            # Load image (path or URL)
            img_path = self.image_path
            log(f"Loading image: {img_path}")
            if img_path.startswith('http://') or img_path.startswith('https://'):
                resp = requests.get(img_path, timeout=20)
                resp.raise_for_status()
                from PIL import Image
                from io import BytesIO
                image = Image.open(BytesIO(resp.content)).convert('RGB')
            else:
                image = load_image(img_path.strip())

            log("Preprocessing image…")
            image = self._resizecrop(image, 352, 640)
            image = frame_process(image)[None, :, None, :, :].to(dtype=weight_dtype, device=device)

            # Encode the input image as the first latent
            log("Preparing conditioning latents…")
            padding_video = torch.zeros_like(image).repeat(1, 1, 4 * (self.max_num_output_frames - 1), 1, 1)
            img_cond = torch.concat([image, padding_video], dim=2)
            tiler_kwargs = {"tiled": True, "tile_size": [44, 80], "tile_stride": [23, 38]}
            log("Encoding image with VAE…")
            t0 = time.perf_counter()
            img_cond = vae.encode(img_cond, device=device, **tiler_kwargs).to(device)
            log(f"VAE encode done in {time.perf_counter()-t0:.2f}s")
            mask_cond = torch.ones_like(img_cond)
            mask_cond[:, :, 1:] = 0
            cond_concat = torch.cat([mask_cond[:, :4], img_cond], dim=1)
            log("Encoding visual context…")
            visual_context = vae.clip.encode_video(image)
            log("Sampling initial noise…")
            sampled_noise = torch.randn([1, 16, self.max_num_output_frames, 44, 80], device=device, dtype=weight_dtype)

            conditional_dict = {
                "cond_concat": cond_concat.to(device=device, dtype=weight_dtype),
                "visual_context": visual_context.to(device=device, dtype=weight_dtype)
            }

            # Pre-fill neutral actions tensors with the right length
            num_frames = (self.max_num_output_frames - 1) * 4 + 1
            if self.mode == 'universal':
                keyboard_dim = 4
                mouse_dim = 2
            elif self.mode == 'gta_drive':
                keyboard_dim = 2
                mouse_dim = 2
            else:
                keyboard_dim = 7
                mouse_dim = 0

            if mouse_dim:
                conditional_dict['mouse_cond'] = torch.zeros((1, num_frames, mouse_dim), device=device, dtype=weight_dtype)
            conditional_dict['keyboard_cond'] = torch.zeros((1, num_frames, keyboard_dim), device=device, dtype=weight_dtype)

            def on_frame_cb(frame_rgb: np.ndarray):
                self._on_frame(frame_rgb)

            # Convert any plain list actions from UI into torch tensors on-device
            def action_provider_wrapped(mode_str: str):
                acts = self._action_provider(mode_str)
                out = {}
                for k, v in acts.items():
                    if isinstance(v, list):
                        out[k] = torch.tensor(v, device=device)
                    else:
                        out[k] = v
                return out

            # Run streaming inference without CLI prompts
            log("Starting pipeline.inference()…")
            t0 = time.perf_counter()
            pipeline.inference(
                noise=sampled_noise,
                conditional_dict=conditional_dict,
                return_latents=False,
                output_folder="outputs",
                name=os.path.basename(img_path) if not img_path.startswith('http') else 'web_image',
                mode=self.mode,
                on_frame=on_frame_cb,
                action_provider=action_provider_wrapped,
                interactive=False,
            )
            log(f"pipeline.inference() finished in {time.perf_counter()-t0:.2f}s")

        except torch.cuda.OutOfMemoryError as e:
            msg = "CUDA OOM: reduce MATRIX_MAX_FRAMES or switch MATRIX_DEVICE=cpu"
            self.status.emit(msg)
            print(msg, file=sys.stderr)
            tb = traceback.format_exc()
            print(tb, file=sys.stderr)
            self._had_error = True
        except Exception as e:
            tb = traceback.format_exc()
            # Emit full error to status and print to console for debugging
            self.status.emit(f"Error: {e}")
            print(tb, file=sys.stderr)
            self._had_error = True
        finally:
            self.finished.emit()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Matrix-Game 2.0 — GUI")
        self.setMinimumSize(1024, 640)

        # Global dark theme
        self.setStyleSheet(
            """
            QMainWindow { background: #000; }
            QWidget { background: #000; color: #fff; }
            QScrollArea { background: #000; border: none; }
            QScrollArea > QWidget > QWidget { background: #000; }
            QLabel { color: #fff; }
            QStatusBar { background: #000; color: #888; }
            """
        )

        # Stacked: page 0 = gallery, page 1 = viewer
        self.stack = QtWidgets.QStackedWidget()
        self.viewer = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.viewer.setStyleSheet("background:#000")
        self.gallery = self.build_gallery()
        self.stack.addWidget(self.gallery)
        self.stack.addWidget(self.viewer)
        self.setCentralWidget(self.stack)

        # State
        self.current_action = {"mouse": None, "keyboard": None}
        self.worker_thread = None
        self.worker = None
        # Allow overriding frames via env var without code edits
        try:
            self.max_num_output_frames = int(os.environ.get("MATRIX_MAX_FRAMES", "120"))
        except Exception:
            self.max_num_output_frames = 120
        self.seed_value = 42
        self.pretrained_model_path = "pretrained_model"
        self._last_frame_buf = None  # keep ref to frame memory to avoid GC with QImage
        self._had_error = False

        # True fullscreen by default for immersive grid
        self.showFullScreen()

    def set_action(self, key: str):
        mode = self.running_mode
        if mode == 'universal':
            keyboard_map = {'w':[1,0,0,0], 's':[0,1,0,0], 'a':[0,0,1,0], 'd':[0,0,0,1], 'q':[0,0,0,0]}
            mouse_map = {'a':[0,-0.1], 'd':[0,0.1], 'q':[0,0], 'w':[0,0], 's':[0,0]}
            self.current_action['keyboard'] = keyboard_map.get(key, [0,0,0,0])
            self.current_action['mouse'] = mouse_map.get(key, [0,0])
        elif mode == 'gta_drive':
            keyboard_map = {'w':[1,0], 's':[0,1], 'q':[0,0], 'a':[0,0], 'd':[0,0]}
            mouse_map = {'a':[0,-0.1], 'd':[0,0.1], 'q':[0,0], 'w':[0,0], 's':[0,0]}
            self.current_action['keyboard'] = keyboard_map.get(key, [0,0])
            self.current_action['mouse'] = mouse_map.get(key, [0,0])
        else:  # templerun
            keyboard_map = {'w':[0,1,0,0,0,0,0], 's':[0,0,1,0,0,0,0], 'a':[0,0,0,0,0,1,0],
                            'd':[0,0,0,0,0,0,1], 'z':[0,0,0,1,0,0,0], 'c':[0,0,0,0,1,0,0], 'q':[1,0,0,0,0,0,0]}
            self.current_action['keyboard'] = keyboard_map.get(key, [1,0,0,0,0,0,0])
            self.current_action['mouse'] = None
        # No label in minimal UI

    def actions_provider(self, mode: str):
        # Provide the currently selected action, defaulting to 'q' (no move)
        if self.current_action['keyboard'] is None and self.current_action['mouse'] is None:
            self.set_action('q')
        return {k:v for k,v in self.current_action.items() if v is not None}

    @QtCore.Slot(np.ndarray)
    def on_frame(self, frame_rgb: np.ndarray):
        # Keep a contiguous copy to ensure QImage data stays valid
        self._last_frame_buf = np.ascontiguousarray(frame_rgb)
        h, w, _ = self._last_frame_buf.shape
        qimg = QtGui.QImage(self._last_frame_buf.data, w, h, 3*w, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg)
        self.viewer.setPixmap(pix.scaled(self.viewer.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    @QtCore.Slot(str)
    def on_status(self, msg: str):
        self.statusBar().showMessage(msg)

    # ---------- Minimal Gallery UI ----------
    def build_gallery(self) -> QtWidgets.QWidget:
        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        # Hide scrollbars but keep content scrollable via gestures/keys
        scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet(
            """
            QScrollBar:vertical, QScrollBar:horizontal { width:0px; height:0px; }
            """
        )
        inner = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(inner)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(8)
        self._grid_layout = grid
        self._thumb_buttons = []  # list of (button, pixmap)

        # Presets mapping to config/checkpoint/mode and source image directories
        self.presets = [
            {"label": "Universal (base)", "mode": "universal",
             "config": "configs/inference_yaml/inference_universal.yaml",
             "checkpoint": "pretrained_model/base_model/diffusion_pytorch_model.safetensors",
             "img_dir": "demo_images/universal"},
            {"label": "Universal (distilled)", "mode": "universal",
             "config": "configs/inference_yaml/inference_universal.yaml",
             "checkpoint": "pretrained_model/base_distilled_model/base_distill.safetensors",
             "img_dir": "demo_images/universal"},
            {"label": "GTA Drive", "mode": "gta_drive",
             "config": "configs/inference_yaml/inference_gta_drive.yaml",
             "checkpoint": "pretrained_model/gta_distilled_model/gta_keyboard2dim.safetensors",
             "img_dir": "demo_images/gta_drive"},
            {"label": "Temple Run", "mode": "templerun",
             "config": "configs/inference_yaml/inference_templerun.yaml",
             "checkpoint": "pretrained_model/templerun_distilled_model/templerun_7dim_onlykey.safetensors",
             "img_dir": "demo_images/temple_run"},
        ]

        # Build a flat grid: each preset contributes its images (universal duplicated for distilled)
        row = col = 0
        self.max_cols = 4
        thumb_size = QtCore.QSize(360, 200)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        for preset in self.presets:
            img_root = os.path.join(base_dir, preset["img_dir"])  # resolve relative to this file
            if not os.path.isdir(img_root):
                self.on_status(f"Missing images folder: {img_root}")
                continue
            for fname in sorted(os.listdir(img_root)):
                if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                fpath = os.path.join(img_root, fname)
                w = self._make_thumb_widget(fpath, preset, thumb_size)
                grid.addWidget(w, row, col)
                col += 1
                if col >= self.max_cols:
                    col = 0
                    row += 1

        inner.setLayout(grid)
        scroll.setWidget(inner)
        layout.addWidget(scroll)
        # Initial sizing
        QtCore.QTimer.singleShot(0, self.update_thumb_sizes)
        return container

        
    def _make_thumb_widget(self, image_path: str, preset: dict, size: QtCore.QSize) -> QtWidgets.QWidget:
        frame = QtWidgets.QFrame()
        frame.setFrameShape(QtWidgets.QFrame.Box)
        frame.setLineWidth(1)
        frame.setStyleSheet("QFrame { background: #000; border: 1px solid #111; }")
        v = QtWidgets.QVBoxLayout(frame)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(4)

        # Image button
        pix = QtGui.QPixmap(image_path)
        btn = QtWidgets.QPushButton()
        btn.setCursor(QtCore.Qt.PointingHandCursor)
        btn.setStyleSheet("QPushButton { border: none; background:#000; } QPushButton:pressed { opacity: 0.9; }")
        btn.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        if not pix.isNull():
            icon = QtGui.QIcon(pix)
            btn.setIcon(icon)
            btn.setIconSize(size)
        else:
            btn.setText(os.path.basename(image_path))
            btn.setStyleSheet("QPushButton { color:#444; border: 1px dashed #333; background:#000; }")
            self.on_status(f"Could not load: {image_path}")
        btn.clicked.connect(lambda: self.start_generation_auto(preset, image_path))
        v.addWidget(btn)

        # Keep for responsive sizing
        self._thumb_buttons.append((btn, pix))
        return frame

    def update_thumb_sizes(self):
        if not hasattr(self, "_thumb_buttons") or not self._thumb_buttons:
            return
        # Compute tile size to fill width with max_cols and spacing
        scroll = None
        # Find the scroll area in the gallery
        for child in self.gallery.findChildren(QtWidgets.QScrollArea):
            scroll = child
            break
        if scroll is None:
            return
        spacing = self._grid_layout.horizontalSpacing() or 0
        margins = self._grid_layout.contentsMargins()
        avail_w = scroll.viewport().width() - margins.left() - margins.right()
        tile_w = max(80, int((avail_w - (self.max_cols - 1) * spacing) / self.max_cols))
        tile_h = int(tile_w * 9 / 16)  # 16:9 cells
        size = QtCore.QSize(tile_w, tile_h)
        for btn, pix in self._thumb_buttons:
            btn.setMinimumSize(size)
            btn.setMaximumSize(size)
            btn.setIconSize(size)

    # ---------- Run control ----------
    def start_generation_auto(self, preset: dict, image_path: str):
        if self.worker_thread is not None:
            self.on_status("Already running")
            return
        # Derived config from preset
        self.running_mode = preset['mode']
        config_path = preset['config']
        checkpoint_path = preset['checkpoint']
        pretrained_model_path = self.pretrained_model_path

        # Validate existence lightly
        if not os.path.exists(config_path):
            self.on_status(f"Missing config: {config_path}")
            return
        if not os.path.exists(pretrained_model_path):
            self.on_status("Invalid pretrained_model_path")
            return
        if not os.path.exists(checkpoint_path):
            self.on_status(f"Missing checkpoint: {checkpoint_path}")
            return

        # Switch to viewer page and show the clicked image immediately as a placeholder
        self.stack.setCurrentWidget(self.viewer)
        try:
            preview_pix = QtGui.QPixmap(image_path)
            if not preview_pix.isNull():
                self.viewer.setPixmap(preview_pix.scaled(self.viewer.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        except Exception:
            pass

        self._had_error = False
        self.worker_thread = QtCore.QThread()
        # Allow overriding device via env var (auto|cuda|cpu)
        device_env = os.environ.get("MATRIX_DEVICE", "auto").lower()
        if device_env not in ("auto", "cuda", "cpu"):
            device_env = "auto"

        self.worker = GenerationWorker(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            pretrained_model_path=pretrained_model_path,
            mode=self.running_mode,
            seed=self.seed_value,
            max_num_output_frames=self.max_num_output_frames,
            device=device_env,
            actions_provider=self.actions_provider,
            image_path=image_path,
        )
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.frameReady.connect(self.on_frame)
        self.worker.status.connect(self.on_status)
        self.worker.finished.connect(self.cleanup_worker)
        self.worker_thread.start()
        self.on_status("Running… (Esc to stop)")

    @QtCore.Slot()
    def stop_generation(self):
        if self.worker is not None:
            self.worker.stop()
        self.on_status("Stopping…")

    @QtCore.Slot()
    def cleanup_worker(self):
        if self.worker_thread is not None:
            self.worker_thread.quit()
            self.worker_thread.wait()
        self.worker_thread = None
        self.worker = None
        # Stay on viewer; allow user to go back with Backspace. Show status.
        if self._had_error:
            self.on_status("Run ended with error. Press Backspace to return to gallery.")
        else:
            self.on_status("Run finished. Press Backspace to return to gallery.")

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        key = event.key()
        if key == QtCore.Qt.Key_Escape:
            self.stop_generation()
        elif key == QtCore.Qt.Key_Backspace and self.worker_thread is None:
            self.stack.setCurrentWidget(self.gallery)
            self.on_status("Idle — Select an image")
        elif self.stack.currentWidget() is self.viewer:
            if key == QtCore.Qt.Key_W:
                self.set_action('w')
            elif key == QtCore.Qt.Key_S:
                self.set_action('s')
            elif key == QtCore.Qt.Key_A:
                self.set_action('a')
            elif key == QtCore.Qt.Key_D:
                self.set_action('d')
            elif key == QtCore.Qt.Key_Q:
                self.set_action('q')
        # Update grid sizes on resize keys like fullscreen toggles can be handled in resizeEvent
        return super().keyPressEvent(event)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        self.update_thumb_sizes()
        return super().resizeEvent(event)


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
