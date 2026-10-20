from autodistill_grounded_sam import GroundedSAM
from autodistill.detection import CaptionOntology
from autodistill.detection.detection_base_model import NmsSetting
from pathlib import Path
import shutil

base_dir = Path(__file__).resolve().parent
output_dir = base_dir / "labeled_dataset"

# 1. Define your ontology: map exact text prompts to your custom YOLO classes
ontology = CaptionOntology({
    "African buffalo": "buffalo",
    "African elephant": "elephant",
    "rhinoceros": "rhino",
    "zebra": "zebra",
})

# 2. Initialize GroundedSAM
base_model = GroundedSAM(
    ontology=ontology,
    box_threshold=0.50,
    text_threshold=0.40,
)

# 3. Auto-label the entire directory of raw images
# This automatically creates a dataset folder with images/ and labels/ in YOLO format
dataset = base_model.label(
    input_folder=str(base_dir),
    extension=".jpg",
    output_folder=str(output_dir),
    nms_settings=NmsSetting.CLASS_AGNOSTIC,
)


def convert_polygons_to_boxes(labels_dir: Path) -> None:
    """Convert YOLO segmentation labels into normalized YOLO detection labels."""
    for label_directory in labels_dir.rglob("labels"):
        for label_path in label_directory.glob("*.txt"):
            converted_lines = []
            for line_number, line in enumerate(
                label_path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                values = line.split()
                if not values:
                    continue

                if len(values) == 5:
                    converted_lines.append(" ".join(values))
                    continue

                if len(values) < 7 or (len(values) - 1) % 2 != 0:
                    raise ValueError(
                        f"Invalid YOLO annotation in {label_path} on line {line_number}"
                    )

                coordinates = [float(value) for value in values[1:]]
                x_coordinates = coordinates[0::2]
                y_coordinates = coordinates[1::2]
                x_min, x_max = min(x_coordinates), max(x_coordinates)
                y_min, y_max = min(y_coordinates), max(y_coordinates)
                center_x = (x_min + x_max) / 2
                center_y = (y_min + y_max) / 2
                width = x_max - x_min
                height = y_max - y_min

                converted_lines.append(
                    f"{values[0]} {center_x:.6f} {center_y:.6f} "
                    f"{width:.6f} {height:.6f}"
                )

            label_path.write_text(
                "\n".join(converted_lines) + ("\n" if converted_lines else ""),
                encoding="utf-8",
            )


def organize_dataset(dataset_dir: Path) -> None:
    """Keep all images and labels in one directory without a validation split."""
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    images_dir.mkdir(exist_ok=True)
    labels_dir.mkdir(exist_ok=True)

    for split_dir in (dataset_dir / "train", dataset_dir / "valid"):
        for source_dir_name, target_dir in (("images", images_dir), ("labels", labels_dir)):
            source_dir = split_dir / source_dir_name
            if not source_dir.exists():
                continue
            for source_path in source_dir.iterdir():
                target_path = target_dir / source_path.name
                if target_path.exists():
                    target_path.unlink()
                shutil.move(str(source_path), str(target_path))

    for temporary_dir in (
        dataset_dir / "annotations",
        dataset_dir / "train",
        dataset_dir / "valid",
    ):
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)

    (dataset_dir / "data.yaml").write_text(
        "names:\n"
        "- buffalo\n"
        "- elephant\n"
        "- rhino\n"
        "- zebra\n"
        "nc: 4\n"
        "train: images\n",
        encoding="utf-8",
    )


convert_polygons_to_boxes(output_dir)
organize_dataset(output_dir)
print("Auto-labeling complete. Dataset ready in YOLO format.")