from typing import Literal

import numpy as np
import numpy.typing as npt

SizeType = Literal[320, 640]
BoxesType = npt.NDArray[npt.NDArray[np.int32]]
ScoresType = npt.NDArray[np.float32]
