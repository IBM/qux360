import logging

# Configure logging defaults for the library
# Libraries should be quiet by default - users control verbosity in their code
logging.getLogger('pyqual').setLevel(logging.WARNING)

# Suppress Mellea's FancyLogger (our LLM orchestration dependency)
# Mellea is extremely verbose by default (INFO level). Since it's an internal
# dependency users don't interact with directly, we suppress it here.
# Users can still enable for debugging: logging.getLogger('fancy_logger').setLevel(logging.DEBUG)
# Note: MelleaSession resets this to DEBUG, so examples re-set it after instantiation
logging.getLogger('fancy_logger').setLevel(logging.WARNING)

from .iffy import IffyIndex
from .validated import Validated, ValidatedList
from .interview import Interview
from .study import Study
from .models import Topic, TopicList, Quote

__all__ = [
    "IffyIndex",
    "Validated",
    "ValidatedList",
    "Interview",
    "Study",
    "Topic",
    "TopicList",
    "Quote",
]
