# Optional Fiji launcher

For full installation, analysis, and troubleshooting instructions, see the [NerveNuc user guide](../README.md#10-use-nervenuc-from-fiji).

The standalone application does not require Fiji. This script adds a Fiji entry point into the same desktop application and processing engine.

1. Install NerveNuc with GUI dependencies in your Python environment (`python -m pip install -e ".[gui]"` from this repository).
2. Copy `Open_in_NerveNuc.groovy` into `Fiji.app/scripts/Plugins/NerveNuc/`, then restart Fiji.
3. Open an image in Fiji and run the NerveNuc script. Choose the Python executable for your installed environment.

The script exports a TIFF copy of the active image/stack and launches NerveNuc with that copy already open. It does not modify the Fiji image. Processing and results stay in NerveNuc; this launcher does not embed the Qt interface inside Fiji or automatically import results back. TIFF copies and launch logs are kept in the system temporary directory, with paths printed in Fiji's log.

The script is provided for the Groovy scripting runtime bundled in Fiji. Live launch remains unverified in this sandbox because Windows blocks external Fiji processes.
