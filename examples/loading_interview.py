from pathlib import Path
from pyqual.core.interview import Interview

DATA_DIR = Path(__file__).parent / "data"

file =DATA_DIR / "P5.xlsx"
i = Interview(file)

# see what we loaded
i.show(10)

# look at the speakers
print(i.get_speakers())

