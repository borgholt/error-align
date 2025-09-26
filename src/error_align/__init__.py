from error_align.error_align import ErrorAlign as ErrorAlign
from error_align.func import error_align as error_align

try:
    from error_align import baselines as baselines
except ImportError:
    pass
