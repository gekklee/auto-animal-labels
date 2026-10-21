# Automatic Animal Labeling

This project uses Grounded SAM to detect four animals in `.jpg` images and
write YOLO bounding-box labels:

| ID | Class |
| --- | --- |
| 0 | buffalo |
| 1 | elephant |
| 2 | rhino |
| 3 | zebra |

## Open the project in VS Code

1. Clone this repo onto your computer.
2. Add the `.jpg` images you want to label in that folder.
3. Open VS Code.
4. Select **File > Open Folder...** and choose your new project folder.
5. Open `autolabel.py`.

## Create a virtual environment in VS Code

You can create the virtual environment using the VS Code interface:

1. Open the Command Palette with **Ctrl+Shift+P**.
2. Search for and select **Python: Create Environment**.
3. Select **Venv** when VS Code asks for the environment type.
4. Select the Python 3.10 interpreter, or another compatible interpreter.

VS Code creates a `.venv` folder in the project and normally selects it
automatically. If it does not, select it manually:

1. Open the Command Palette with **Ctrl+Shift+P**.
2. Select **Python: Select Interpreter**.
3. Choose the interpreter whose path ends with:

   ```text
   .venv\Scripts\python.exe
   ```

Open a new terminal with **Terminal > New Terminal**. VS Code should activate
the environment automatically. The terminal prompt usually starts with
`(.venv)`.

The output should point to the `.venv\Scripts\python.exe` inside the project
folder.

## Install dependencies

With the virtual environment active, install the packages used by the script:

```powershell
python -m pip install --upgrade pip
python -m pip install -r .\requirements.txt
```

The versions are pinned to the known-working Windows GPU environment. This
uses CUDA 12.4 wheels, so an NVIDIA GPU and a compatible NVIDIA driver are
required. If you do not need GPU acceleration, replace the Torch entries with
a matching CPU-only pair from the [official PyTorch selector](https://pytorch.org/get-started/locally/).

## Add images and run labeling

Put the source `.jpg` images directly beside `autolabel.py` in the project
folder:

```text
animal-labeling/
├── autolabel.py
├── img_0001.jpg
├── img_0002.jpg
└── ...
```

Run the script from the project folder:

```powershell
python .\autolabel.py
```

The output is written to:

```text
labeled_dataset/
├── data.yaml
├── images/
└── labels/
```

## Adjusting detection behavior

The script currently uses:

```python
box_threshold=0.50
text_threshold=0.40
```

Higher values reduce weak detections but can miss animals. If too many false
detections remain, increase the thresholds gradually. If real animals are
missed, reduce them gradually.

The prompts and class IDs are defined in the `ontology` dictionary.

## Troubleshooting

Check that the VS Code terminal is using the virtual environment:

```powershell
python -c "import sys; print(sys.executable)"
```

The output should end with:

```text
\.venv\Scripts\python.exe
```

Check that the main packages are installed and consistent:

```powershell
python -m pip check
python -m pip show autodistill-grounded-sam autodistill supervision roboflow scikit-learn torch torchvision
```

If importing Torch fails with `WinError 1114` and a reference to `c10.dll`,
check that the NVIDIA driver is installed and then reinstall the pinned
environment:

```powershell
python -m pip install --force-reinstall -r .\requirements.txt
```

If a previous run produced poor labels, run the script again after changing
the prompts or thresholds. Review a sample of the files in
`labeled_dataset\labels` before training a model.
