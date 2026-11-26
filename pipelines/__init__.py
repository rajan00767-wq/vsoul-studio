import os
import sys
from pathlib import Path

# Add NVIDIA CUDA DLL search paths on Windows so ONNX Runtime can find CUDA 13/12 libraries
if sys.platform == "win32":
    _site_packages = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    if _site_packages.exists():
        for _p in list(_site_packages.glob("**/bin/x86_64")) + list(_site_packages.glob("**/bin")):
            if _p.is_dir():
                try:
                    os.add_dll_directory(str(_p))
                except Exception:
                    pass
                os.environ["PATH"] = str(_p) + os.pathsep + os.environ.get("PATH", "")

# Ensure InsightFace models always use the workspace directory instead of ~/.insightface
# to prevent PermissionError: [WinError 5] Access is denied: 'C:\Users\rajan\.insightface\models'
_INSIGHTFACE_WORKSPACE_ROOT = str((Path(__file__).resolve().parent.parent / "models" / "insightface").resolve())

try:
    import insightface.app.face_analysis
    _FACE_ANALYSIS_CACHE = {}
    _orig_face_analysis_new = insightface.app.face_analysis.FaceAnalysis.__new__
    _orig_face_analysis_init = insightface.app.face_analysis.FaceAnalysis.__init__

    def _safe_face_analysis_new(cls, *args, **kwargs):
        name = kwargs.get("name", args[0] if args else "buffalo_l")
        providers = tuple(kwargs.get("providers", ()))
        key = (name, providers)
        if key in _FACE_ANALYSIS_CACHE:
            return _FACE_ANALYSIS_CACHE[key]
        inst = _orig_face_analysis_new(cls)
        _FACE_ANALYSIS_CACHE[key] = inst
        return inst

    def _safe_face_analysis_init(self, *args, **kwargs):
        if getattr(self, "_already_initialized", False):
            return
        if "root" not in kwargs or kwargs["root"] in ("~/.insightface", None):
            kwargs["root"] = _INSIGHTFACE_WORKSPACE_ROOT
        _orig_face_analysis_init(self, *args, **kwargs)
        self._already_initialized = True

    insightface.app.face_analysis.FaceAnalysis.__new__ = _safe_face_analysis_new
    insightface.app.face_analysis.FaceAnalysis.__init__ = _safe_face_analysis_init

    import insightface.utils.storage
    _orig_download = insightface.utils.storage.download

    def _safe_download(sub_dir, name, force=False, root='~/.insightface'):
        if root in ('~/.insightface', None):
            root = _INSIGHTFACE_WORKSPACE_ROOT
        return _orig_download(sub_dir, name, force=force, root=root)

    insightface.utils.storage.download = _safe_download
except Exception:
    pass
