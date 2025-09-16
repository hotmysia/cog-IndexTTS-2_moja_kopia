__version__ = "1.0.0"

# preserved here for legacy reasons
__model_version__ = "latest"

try:
    import audiotools  # type: ignore
except ImportError:  # pragma: no cover
    audiotools = None

if audiotools is not None:
    audiotools.ml.BaseModel.INTERN += ["dac.**"]
    audiotools.ml.BaseModel.EXTERN += ["einops"]

    from . import nn  # noqa: F401
    from . import model  # noqa: F401
    from . import utils  # noqa: F401
    from .model import DAC  # noqa: F401
    from .model import DACFile  # noqa: F401
else:
    class _MissingAudiotools:  # pragma: no cover
        def __getattr__(self, name):
            raise ImportError(
                "descript-audiotools is required for DAC components but is not installed."
            )

    audiotools = _MissingAudiotools()  # type: ignore
    DAC = None  # type: ignore
    DACFile = None  # type: ignore

__all__ = ["DAC", "DACFile"]
