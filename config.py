import pathlib

GROUND_TRUTH_PATH = pathlib.Path(__file__).parent.resolve() / "ground_truth"
EVALUATION_PATH = pathlib.Path(__file__).parent.resolve() / "results" / "evaluation"
VISUALIZATION_PATH = pathlib.Path(__file__).parent.resolve() / "results" / "visualization"
SEGMENTATION_PATH = pathlib.Path(__file__).parent.resolve() / "results" / "binarizations"
RECONSTRUCTION_PATH = pathlib.Path(__file__).parent.resolve() / "results" / "reconstructions"