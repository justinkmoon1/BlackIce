from Processor import Processor

ANNOT_PATH = "Test Set Black Final/_annotations.coco.json"

NEW_ANNOT_PATH = "Test Set Augmentation"

IMAGE_DIR = "Test Set Black Final/train"

NEW_IMAGE_DIR = "Test Set Augmentation/train"

FILE_NAME = "_annotations.coco.json"


processor = Processor()

processor.random_selection(ANNOT_PATH, NEW_ANNOT_PATH, IMAGE_DIR, NEW_IMAGE_DIR, 3, FILE_NAME)

