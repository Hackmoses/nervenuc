# NerveNuc — step-by-step user guide

NerveNuc is a desktop application for detecting and counting nuclei/objects, measuring fluorescence inside detected objects, and classifying marker positivity. Its default detection approach is designed for fluorescence nuclear images, including DAPI-anchored microscopy.

The everyday workflow is **Open → Set up → Analyze → Results**. Single-image analysis and folder processing do not require animal IDs, experimental groups, or mandatory manual review.

| How you want to work | What to use |
|---|---|
| Analyze images directly on your computer | The standalone application. Fiji is optional. |
| Start from an image already open in Fiji | The included **Open in NerveNuc** Groovy launcher, which opens the same application with a TIFF copy of Fiji's active image. |

The Fiji launcher does not embed the interface inside Fiji or automatically return results to Fiji. Install the standalone application first.

## Contents

- [1. Requirements](#1-requirements)
- [2. Install on a Windows PC](#2-install-on-a-windows-pc)
- [3. Launch again or make a desktop launcher](#3-launch-again-or-make-a-desktop-launcher)
- [4. Prepare your images](#4-prepare-your-images)
- [5. Analyze your first image](#5-analyze-your-first-image)
- [6. Optional preprocessing](#6-optional-preprocessing)
- [7. Review and marker positivity](#7-review-and-marker-positivity)
- [8. Export and understand results](#8-export-and-understand-results)
- [9. Process a folder](#9-process-a-folder)
- [10. Use NerveNuc from Fiji](#10-use-nervenuc-from-fiji)
- [11. Open a NerveNuc image in Fiji](#11-open-a-nervenuc-image-in-fiji)
- [12. Optional custom and study tools](#12-optional-custom-and-study-tools)
- [13. Troubleshooting](#13-troubleshooting)
- [14. Scientific interpretation](#14-scientific-interpretation)
- [15. Updates and developer checks](#15-updates-and-developer-checks)

## 1. Requirements

For standalone use:

- **64-bit Python 3.11 or newer**. The main instructions below use Windows and PowerShell.
- The NerveNuc source folder containing `pyproject.toml`, `src`, and `tests`. Obtain it from the maintainer and extract any source archive before installation.
- Internet access for the first installation, to download dependencies.
- Storage for input images and a separate results folder. Large Z-stacks require more memory and processing time than 2D images.

This guide describes a source installation; Python is currently required. It does not assume a bundled Windows installer. Once installed, the interface can be used without writing code.

For the optional launcher, also install [Fiji from its official download page](https://imagej.net/software/fiji/downloads). The script uses Fiji's Groovy runtime. **PyImageJ and an in-process Python/Java bridge are not required for this launcher.**

The desktop workflow has been exercised on Windows. Equivalent macOS/Linux commands are included below, but those platforms have not received the same testing.

## 2. Install on a Windows PC

### Step 1 — Install Python

1. Download a 64-bit installation from [Python's official Windows downloads](https://www.python.org/downloads/windows/).
2. Complete installation. Enable an option to add Python to PATH if your installer offers it.
3. Open a **new PowerShell window**.
4. Check Python:

```powershell
py -3 --version
```

You should see Python 3.11 or newer. If `py` is unavailable but `python` works, use `python` instead of `py -3` when creating the environment below.

### Step 2 — Locate the project folder

1. Copy or extract the project to a stable location, such as `C:\Projects\NerveNuc`.
2. Confirm `pyproject.toml` is directly in that folder. Use the inner folder if your archive created an extra outer directory.
3. Open PowerShell in that folder, or run:

```powershell
cd "C:\Projects\NerveNuc"
```

Replace the example path with yours. Keep quotation marks around paths containing spaces.

### Step 3 — Create a private Python environment

```powershell
py -3 -m venv .venv
```

The `.venv` folder keeps NerveNuc's packages separate from other software. These instructions call its Python directly, so **you do not need environment activation or a PowerShell execution-policy change**. See [Python's virtual-environment documentation](https://docs.python.org/3/library/venv.html).

### Step 4 — Install the application

Run in the folder containing `pyproject.toml`:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install -e ".[gui]"
```

Wait for installation to finish without errors. `[gui]` includes the desktop dependency; `-e` links the installation to this source folder. Keep the folder in place after installation.

### Step 5 — Launch

```powershell
.\.venv\Scripts\python.exe -m nervenuc.gui.app
```

The window should display **Open**, **Set up**, **Analyze**, and **Results** on the left. If it opens, continue to Section 5. If it fails, use Troubleshooting before adding Fiji or other packages.

### Optional — macOS/Linux commands

From a terminal in the project folder, using Python 3.11 or newer:

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip setuptools wheel
./.venv/bin/python -m pip install -e ".[gui]"
./.venv/bin/python -m nervenuc.gui.app
```

Linux may additionally require its system Qt/display libraries; use the actual error message to diagnose distribution-specific requirements.

## 3. Launch again or make a desktop launcher

Installation is a one-time step. On later visits:

```powershell
cd "C:\Projects\NerveNuc"
.\.venv\Scripts\python.exe -m nervenuc.gui.app
```

Alternatively, use the installed command:

```powershell
.\.venv\Scripts\nervenuc-gui.exe
```

To start with a specific image:

```powershell
.\.venv\Scripts\python.exe -m nervenuc.gui.app "C:\Microscopy\sample.ome.tif"
```

### Optional — Double-click launcher

1. Open Notepad.
2. Paste this, replacing the project path with yours:

```bat
@echo off
cd /d "C:\Projects\NerveNuc"
".venv\Scripts\python.exe" -m nervenuc.gui.app
pause
```

3. Save as `Start-NerveNuc.bat`, choosing **All files** so it does not become `.bat.txt`.
4. Double-click it. After closing the app, press a key to close the console. The console can show startup errors.

If you move the project or recreate `.venv`, update your launchers and installation paths.

## 4. Prepare your images

| Input | Use and interpretation |
|---|---|
| TIFF / OME-TIFF (`.tif`, `.tiff`) | Preferred for scientific fluorescence images, channels, and Z-stacks. OME-TIFF can preserve dimension and calibration metadata. |
| PNG / JPEG / BMP | Supported as 2D images. RGB channels are color components and may not correspond to independently acquired fluorescence markers. |
| Confocal / widefield | Supported when saved in an accepted format with interpretable channels/dimensions. Acquisition method is different from file format. |
| Brightfield | Can be loaded, but the bright-object fluorescence detector may need manual settings or may be unsuitable for the specimen. |
| Proprietary microscope formats | Not read directly. Use suitable acquisition software or a Fiji importer, then export TIFF/OME-TIFF while preserving channels/dimensions. |

For quantitative fluorescence, use original acquisition data where possible. Avoid screenshots, flattened color composites, JPEG compression, and display-adjusted exports as substitutes for raw signal channels. Keep original images separate from results.

Start with one representative image and inspect its detections before processing a folder.

## 5. Analyze your first image

### Step 1 — Open

1. Select **Open** on the left.
2. Click **Choose File...**, or use **File → Open Image/Stack...** (`Ctrl+O`).
3. Select an image and wait for its background loading to finish.
4. Inspect the preview and dimensions.
5. For Z-stacks, inspect slices with the slider and previous/next buttons.
6. Adjust channel visibility, color, and gamma to see structures clearly. These display controls do not change raw measurements.
7. Use **Fit to View** to center the image. The mouse wheel zooms; dragging pans.

If dimensions are ambiguous, the app may ask whether pages represent Z slices, channels, or mixed Z/channel ordering. Confirm the acquisition's ordering rather than guessing. Metadata-preserving OME-TIFF is preferable when available.

Opening another image resets image-specific channel mappings, cutoffs, custom metrics, and slice settings. Check the new setup.

### Step 2 — Choose the mode and task

Under **Image setup**, choose:

| Mode | When to use it | Interpretation |
|---|---|---|
| **Analyze this 2D image** | A 2D image without a Z-stack. | Objects and measurements in an XY plane. |
| **Analyze a stack projection** | You want a 2D analysis of Z-stack data; the default for a stack. | Measurements on a projected XY image. Different Z objects can overlap in projection. |
| **Analyze the full stack in 3D** | You need volumetric detection. | Measurements inside 3D masks, rather than a projection. |

Choose a **Task**:

| Task | Required channels | Result |
|---|---|---|
| **Count objects** | One object channel; no separate signal required. | Count, object geometry, and masks. |
| **Count and measure signal** | One object channel and at least one signal channel. | Raw marker statistics inside detected objects. |
| **Count and classify marker positivity** | One object channel and at least one signal channel. | Raw measurements first, then positive/negative calls after explicit cutoff application. |

### Assign channels

In **Channel Assignment**, give each channel a role:

- **Objects to count:** defines the objects/nuclei to detect. Assign exactly one channel.
- **Signal to measure:** a marker to quantify inside those masks. Give each signal a nonempty, unique name.
- **Ignore:** excludes a channel from this analysis.

Example:

| Channel | Role | Signal name |
|---|---|---|
| DAPI | Objects to count | Not required |
| NeuN | Signal to measure | `NeuN` |
| An unused stain | Ignore | Not required |

Channel 0 is not necessarily DAPI. Confirm metadata and preview against your acquisition.

### Check stack/time options

1. **Z Start Slice** and **Z End Slice** are 1-based and inclusive: 1–10 includes slices 1 through 10.
2. The default is the full stack. Use a narrower range only when appropriate for the experiment.
3. For projected 2D, choose `max` or `mean` under **Projection Method**. `max` keeps the brightest value at each XY position; `mean` averages the selected slices.
4. Full 3D analyzes the selected volume; its projection-method control does not turn the analysis into a projection.
5. If **Time frame** appears, select the frame to analyze. The quick workflow does not automatically track objects through all time points.

Click **Continue to Analyze**.

### Step 3 — Analyze

1. Begin with default settings on your representative image.
2. Set **Minimum object area (pixels)** for 2D or **Minimum object volume (voxels)** for 3D. Smaller objects are removed.
3. Optionally configure [preprocessing](#6-optional-preprocessing).
4. Open **Advanced detection settings** only when needed. The current interface offers automatic `otsu` or `manual` thresholding, manual cutoff, hole filling, and border exclusion.
5. Click **Analyze image**.
6. Wait for completion. Configuration controls are temporarily disabled to keep the run consistent.
7. The app moves to Results with the detected count and measurements for the chosen task.

There is no separate preparation button, no required animal/group metadata, and no required review-finalization step in this quick workflow.

If detections are poor, inspect the overlay, adjust settings, and rerun. Check representative masks and counts before interpreting results biologically.

### Step 4 — Inspect Results

1. Inspect counts and object outlines.
2. Choose **Objects**, **Signal measurements**, **Positive / negative calls**, or **Custom measurements** in the results-table selector. Some tables remain empty until the associated task/calculation is performed.
3. Use **Show all columns** to inspect additional IDs and provenance. The default view has a smaller set of readable columns.
4. Optionally review detections or apply positivity cutoffs as described below.
5. Export when the results are current.

## 6. Optional preprocessing

Open **Analyze → Preprocessing options**. These operations affect a detection copy of the object channel. **Raw marker measurements and original source pixels remain unchanged.** In projected 2D, measurements use the unprocessed marker projections.

| Option | Purpose | Starting point |
|---|---|---|
| **Subtract fluorescence background** | Rolling-ball estimation removes diffuse detection baseline. | Off by default. Inspect feature retention and background reduction after choosing a radius. |
| **Mild Gaussian denoising** | Reduces small-scale noise before detection. | On by default, sigma 1 pixel. Heavy blur can merge objects or weaken boundaries. |
| **Estimate uneven illumination** | Estimates broad shading from the image itself. | Off by default. Use a scale substantially larger than structures being detected. |
| **Scale detection intensity to 0–1** | Scales using the detection image's 1st–99th percentile range. | Off by default. Changes detection cutoff units, not raw marker measurement units. |

Order: **background subtraction → Gaussian denoising → estimated illumination correction → optional scaling**. Disabled operations are skipped.

Processing happens after spatial preparation: on the projection for projected 2D, or independently inside each XY plane for true 3D. Gaussian denoising does not blur across Z. Intensity scaling uses one shared range across the selected stack.

1. Choose background/illumination scales larger than features you intend to preserve. Defaults are starting points, not universal validated settings.
2. Run Analyze.
3. Toggle **Show processed detection image** to inspect the processed object channel, then toggle back to compare with raw channels.
4. With scaling enabled, the manual **detection** cutoff changes to 0–1 units. Marker positivity still uses raw measured signal.
5. Rerun Analyze after processing settings change.

This illumination estimate is not calibrated flat-field correction. Detection scaling does not make different exposure, laser power, gain, staining, or saturated acquisitions quantitatively comparable. See [scikit-image's rolling-ball documentation](https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_rolling_ball.html) and [ImageJ intensity-processing guidance](https://imagej.net/mbf/image_intensity_proce.htm).

## 7. Review and marker positivity

### Optional — Reject or restore detections

1. In Results, click **Review / edit detections...**, or use **Advanced → Review detections**.
2. Select an object row and inspect its location and overlay.
3. Click **Reject (R)** to exclude an incorrect detection. With focus in the review widget, `R` is its shortcut.
4. Counts and measurements update immediately. Old classifications and custom summaries clear because the population changed.
5. To reverse a rejection, select its rejected row and click **Restore (U)**. Selection follows the object when its row moves, allowing immediate restoration. `U` is the restore shortcut within the review widget.
6. Click **Back to image results** when done.

Rejected objects do not contribute to current measurements. Reapply cutoffs after editing detections if you need updated classifications.

The quick review panel supports rejection/restoration, not full freehand segmentation editing. Some separate legacy/programmatic ROI capabilities may be disabled for quick-result sessions.

### Optional — Classify marker positivity

1. Choose **Count and classify marker positivity**, assign signals, and run Analyze.
2. Click **Set positive / negative cutoffs...**, or use **Advanced → Positive / negative cutoffs**.
3. Set each marker's **Threshold Cutoff**. Initial suggestions use the median measured signal: a starting suggestion, not a validated biological threshold.
4. Under **Advanced cutoff options**, optionally choose a statistic such as `mean_intensity`, `median_intensity`, or `integrated_intensity`, and comparison `>=` or `>`.
5. Click **Preview Phenotypes** to inspect positive counts at a proposed cutoff.
6. Set all marker cutoffs before clicking **Apply cutoffs and view results**. This applies the current settings and returns to Results.
7. Inspect positive/negative calls, counts, and percentages, then export.

Use controls and experimental design to justify cutoffs. A median suggestion is not evidence that half the objects should be biologically positive. Detection scaling does not change raw marker measurement units.

## 8. Export and understand results

### Save a single-image analysis

1. Complete Analyze. Apply cutoffs if you need positivity calls.
2. Click **Export...** in Results.
3. Choose a results folder. For a first export, use a new empty folder separate from input images.
4. Wait for the saved-location message.
5. Open CSV tables in Excel, R, Python, or another spreadsheet application. Inspect TIFFs in an image viewer or Fiji.

If NerveNuc results already exist in that folder:

- **Yes:** saves a new numbered subfolder such as `export-001`, preserving originals.
- **No:** lets you choose another folder.
- **Cancel:** stops export.

For stale results, rerun Analyze after detection/preprocessing/channel changes. If only positivity cutoffs/custom metrics changed, apply the updated calculations before exporting.

### Export files

Not every task produces every file.

| File | Contents / availability |
|---|---|
| `objects.csv` | Accepted object records and geometry, including count-only results. |
| `measurements.csv` | Raw per-object marker statistics for signal/positivity tasks. |
| `classification.csv` | Positive/negative calls after explicit cutoff application. |
| `summary.csv` | Optional custom biological metrics after calculation. |
| `labels.tif` | Accepted integer object masks. Background is 0. For 3D, the full analyzed volume is saved. |
| `detection-input.tif` | Detection image/volume after managed preprocessing; not the raw marker measurement image. |
| `annotated-view.png` | Full-resolution annotated XY plane. In 3D, this is the selected Z slice, not the whole volume. |
| `analysis.json` | Executed settings, source identity, preprocessing, count, review history/policy, and annotation image/slice basis. |

Quick exports are CSV tables. Multi-sheet Excel/GraphPad exports belong to the optional study workflow; quick CSVs can still be opened in Excel.

### Read measurements and IDs

- `mean_intensity`: average raw marker signal inside the mask.
- `median_intensity`: median raw marker signal inside the mask.
- `integrated_intensity`: sum of raw marker signal inside the mask; depends on intensity and object size.
- 2D area is in pixels; 3D volume is in voxels. Physical volume needs valid voxel calibration.
- Match `current_label` values to `labels.tif`; use review IDs to follow object history. In 2D, numeric labels can change after edits while review IDs track identity.
- Projected-2D signal measurements describe unprocessed marker projections, not full volumetric signal.

Archive original images, result folders, and settings together. `analysis.json` records an analysis; it is not a saved interactive session that can be resumed by double-clicking it. Export before closing.

## 9. Process a folder

Use this for independent images sharing settings, without animal/group metadata.

1. Open a representative image.
2. Assign channels and choose task/mode.
3. Configure stack, detection, and preprocessing settings.
4. Run Analyze and inspect the result before scaling up.
5. For positivity, establish explicit marker cutoffs on the representative image first.
6. Return to **Analyze** and click **Process a folder with these settings...**.
7. Select the input folder.
8. Select a **different, empty output folder**. Existing runs are protected.
9. Wait for progress/status messages.
10. Open `folder-report.csv` and check every image's status, count, and error message.
11. Inspect per-image subfolders named with an ordinal and filename stem.

Behavior to check:

- Supported images directly in the chosen folder are processed; subfolders are not searched recursively.
- Settings reuse channel indices and Z ranges. Confirm compatible acquisition layouts: Channel 1 must represent the intended marker in every input.
- Do not mix direct 2D images and stacks in a run expecting one mode. Corrupt/incompatible inputs are reported as failures while compatible files can continue.
- A selected Z range must fit each stack; shorter stacks may fail.
- Folder exports include per-image tables, labels, detection input when applicable, settings, and the report. They do not include the single-image annotated PNG or calculate custom study/group summaries.
- Correct failed inputs and rerun separately or into a new empty output folder. Completion does not mean every image succeeded.

## 10. Use NerveNuc from Fiji

### What the plugin does

`fiji-plugin/Open_in_NerveNuc.groovy` uses Fiji's active image, saves a TIFF copy to the system temporary folder, and launches the installed NerveNuc environment with that copy already selected. The original Fiji image remains unchanged.

Analysis happens in NerveNuc's standalone window. Results are not automatically returned to Fiji; export them and open the resulting files in Fiji manually when needed.

### Step 1 — Confirm standalone installation

Complete Section 2 and verify NerveNuc opens. From the project folder, print the correct Python path:

```powershell
.\.venv\Scripts\python.exe -c "import sys; print(sys.executable)"
```

Copy the full path, for example:

```text
C:\Projects\NerveNuc\.venv\Scripts\python.exe
```

Use that environment's executable, not an unrelated system Python and not the `py` launcher.

### Step 2 — Install the Fiji menu command

1. Close Fiji.
2. Locate `Fiji.app`. On macOS, reveal the application package contents if needed.
3. Create `scripts/Plugins/NerveNuc` inside the Fiji installation if it does not exist.
4. Copy the project's `Open_in_NerveNuc.groovy` there:

```text
Fiji.app/
  scripts/
    Plugins/
      NerveNuc/
        Open_in_NerveNuc.groovy
```

5. Keep the filename and `.groovy` extension; make sure it has not become `.groovy.txt`.
6. Restart Fiji.
7. Look for **Plugins → NerveNuc → Open in NerveNuc**. Menu organization can vary with Fiji versions; search for the command if it appears elsewhere.

See [Fiji's official scripting documentation](https://imagej.net/scripting/) for supported extensions and menu-folder registration.

### Step 3 — Launch from your image

1. Open the image/stack in Fiji. For proprietary formats, use a suitable importer preserving channels and dimensions.
2. Make that image window active. An open active image is required.
3. Run **Open in NerveNuc**.
4. Paste the full `.venv` Python path into **Python executable**.
5. **Do not put quotation marks around the path in this dialog**, even when it contains spaces. The script passes it directly as a program path.
6. Click **OK**. Fiji remembers the path after a successful process start.
7. Wait for NerveNuc to open the TIFF copy.
8. Confirm channels, Z/time dimensions, and calibration; follow **Set up → Analyze → Results**.
9. Export to a permanent results folder. Keep the original acquisition as the primary record.

Fiji's Log window prints the scratch TIFF and launch-log paths. A process-start message means Python started; if no NerveNuc window appears, inspect the launch log for the actual error.

The temporary TIFF and log remain in the system temporary directory. Do not rely on that directory for permanent archival.

### Alternative — Run in Fiji's Script Editor

If the menu command does not appear:

1. Open an image in Fiji.
2. Open the **Script Editor**; `[` opens it in supported Fiji versions.
3. Open `Open_in_NerveNuc.groovy` in the editor.
4. Choose **Groovy** as the language if it is not detected.
5. Run the script and enter the installed NerveNuc environment's Python path.

Editor/language instructions are in the [official scripting guide](https://imagej.net/scripting/).

## 11. Open a NerveNuc image in Fiji

For the opposite direction, with NerveNuc already running:

1. Install Fiji if necessary.
2. Choose **Tools → Fiji / ImageJ → Configure Fiji...** in NerveNuc.
3. Select your Fiji installation/executable, validate it in the dialog, and save.
4. Open or analyze an image.
5. Choose **Tools → Fiji / ImageJ → Open Current Image in Fiji**.
6. Inspect it in the external Fiji window.

For completed quick analysis, this sends the prepared object channel, including the selected projection or volume. It does not automatically send every signal channel or import Fiji edits into the quick result.

Open exported `labels.tif` directly in Fiji to inspect masks. Label values are object identifiers; editing that TIFF does not update NerveNuc's measurements.

ROI export/import and split replacement require compatible review sessions and may be disabled in the quick workflow. Do not add bridges solely for normal four-step analysis. If you explicitly need the optional ROI/PyImageJ extras, they can be installed separately:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[gui,fiji]"
```

That extra includes additional dependencies; it is unnecessary for launching NerveNuc from the Groovy script. In-process PyImageJ use has separate Java/environment requirements.

## 12. Optional custom and study tools

### Custom measurements

For a named numerator/denominator metric:

1. Run a positivity analysis and establish meaningful cutoffs.
2. Open **Advanced → Custom measurements**.
3. Name the metric and choose numerator/denominator populations.
4. Add the definition.
5. Return to the cutoff panel and apply the cutoffs/calculations.
6. View **Custom measurements** in Results; export `summary.csv` when available.

Ordinary counts and raw signals need no custom definition.

### Study/batch aggregation

Use **Advanced → Study / batch analysis** only when you explicitly need biological-replicate aggregation rather than independent image-folder exports.

1. Configure or load a compatible **study preset** from Advanced.
2. Click **Add File(s)...**.
3. Use **Edit Sample...** for unique sample IDs, biological replicate/animal IDs, and channel mappings where required.
4. Click **Validate Batch** and resolve problems.
5. Click **Run Batch** after validation enables it.
6. Inspect status, then **Advanced → Study results**.
7. Assign experimental groups where needed for grouped exports.

This separate workflow uses its own preset/configuration and automatically accepts batch detections. Quick preprocessing options are not part of legacy study-preset execution. For those options without study metadata, use **Process a folder**.

Failed samples prevent complete-study export. Successful images are not substitutes for a complete cohort; aggregation and inference require a suitable experimental design.

## 13. Troubleshooting

| Problem | What to do |
|---|---|
| `py` or `python` is not recognized | Install Python and open a new terminal. If only `python` works, use it to create `.venv`. |
| `.venv\Scripts\python.exe` is missing | Check the project directory and create the environment there. `.venv` should be directly inside the project folder. |
| `No module named nervenuc` | Use the correct environment. In the folder containing `pyproject.toml`, rerun `-m pip install -e ".[gui]"` using that environment's Python. |
| `No module named PySide6` | Install the `[gui]` extra into the environment you actually launch. |
| `Cannot import setuptools.build_meta` or missing build tools | Upgrade `pip setuptools wheel` in `.venv` and retry installation. Internet access or an institutional package mirror may be needed. |
| PowerShell blocks environment activation | Use the explicit `.venv\Scripts\python.exe` commands; activation is unnecessary. |
| A dependency has no compatible Python build | Use Python 3.11 or newer with compatible dependency builds. For a very new Python version, a fresh environment using a supported version may be needed. |
| Image dimensions are ambiguous | Confirm channel/Z/time ordering. Prefer a metadata-preserving TIFF/OME-TIFF. |
| A multi-series TIFF is rejected | Export the intended series separately. Explicit series selection is not currently provided in the quick loader. |
| Signal measurement lacks a signal channel | Assign at least one **Signal to measure**, or use **Count objects** for a single-channel image. |
| No objects are detected | Check object channel, size filter, threshold, Z range, and preprocessing. A scaled manual detection cutoff uses 0–1 units. |
| False objects or merged nuclei | Inspect outlines and adjust size, threshold, or processing scales. Strong smoothing can damage separation; reject individual incorrect detections when useful. |
| Export is disabled/stale | Rerun Analyze after upstream changes. Reapply edited cutoffs/custom calculations where appropriate. |
| Positivity disappeared after Reject/Restore | The population changed. Apply cutoffs again for current calls/summaries. |
| Folder output is refused | Choose a different, empty folder. Prior runs are protected. |
| Folder processing finishes with failures | Read `folder-report.csv`; check corrupted files, channel layout, mode, and Z ranges. Rerun corrected inputs separately or in a new output folder. |
| Fiji menu command is missing | Check folder and `.groovy` extension, restart Fiji, or use the Script Editor in Groovy mode. |
| Fiji starts Python but no app appears | Read the launch log printed in Fiji; verify the exact `.venv` Python path and confirm standalone launching works. |
| Fiji reports no current image | Open an image and make its window active before running the script. |
| The Fiji launcher fails after moving the project | Reinstall/recreate the environment and update its full Python path in the dialog. |
| An optional ROI action is disabled | The active workflow may not supply the required review session. Ordinary quick analysis/export does not depend on that action. |
| Large stacks or rolling-ball subtraction are slow | Start with representative data, a suitable Z range, and a reasonable radius. Larger volumes/kernels cost more processing. Wait for the background run before closing. |

For support, report OS, Python and NerveNuc versions, input format/dimensions, task/mode, reproduction steps, and error text. Use a de-identified representative image or synthetic reproduction rather than confidential acquisition data.

## 14. Scientific interpretation

- Automatic acceptance permits a quick workflow; it does not certify detection accuracy.
- Check representative masks/counts against expert review or independently annotated ground truth for your assay.
- Justify fluorescence cutoffs with controls. Saturation and inconsistent acquisition settings can compromise interpretation even when software runs correctly.
- Preprocessing preserves raw marker pixels, but altered detection masks can change which pixels enter a measurement.
- Projection measurements are 2D; they do not establish volumetric overlap or true 3D colocalization.
- Brightfield loading does not guarantee accurate detection of arbitrary structures by the fluorescence-oriented engine.
- Physical volume requires valid calibration; otherwise interpret sizes in voxels.
- Folder outputs are per image. Images/nuclei are not automatically independent biological replicates; define the appropriate experimental unit.
- Quick settings JSON is an analysis record, not a resumable project. Export before closing.

## 15. Updates and developer checks

### Update an existing installation

After obtaining updated source in the same project folder, close NerveNuc and run:

```powershell
cd "C:\Projects\NerveNuc"
.\.venv\Scripts\python.exe -m pip install -e ".[gui]"
.\.venv\Scripts\python.exe -m nervenuc.gui.app
```

If the folder or Python version changes, create a fresh environment and update PC/Fiji launcher paths. Preserve original images and prior export folders when updating.

### Optional developer checks

Ordinary users do not need to run these:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[gui,dev,gui-dev]"
$env:QT_QPA_PLATFORM = "offscreen"
.\.venv\Scripts\python.exe -m pytest -q tests/gui
```

Headless tests check regressions; they do not replace manual interface inspection or scientific validation. Live Fiji and in-process PyImageJ testing have environment/dependency requirements separate from core standalone usage.

### Project files

- `pyproject.toml`: dependencies and the `nervenuc-gui` command.
- `src/nervenuc`: processing engine and desktop interface.
- `fiji-plugin/Open_in_NerveNuc.groovy`: optional Fiji launcher.
- `tests`: regression/integration tests.
- `LICENSE`: software license.
- `CITATION.cff`: citation metadata.

The Python API also provides specialized preparation, segmentation, review, measurement, classification, biological-summary, study-aggregation, and reproducible-export workflows. Start with the four-step desktop workflow unless your study needs those additional tools.
