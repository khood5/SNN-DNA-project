# SNN-DNA-project
This is a project for Kent State Biology Dept. It is designed to read in Single-molecule experiment data and predicted if the experiment (if run for more time) would have a high/medium/low number of events

Use the following commands in your python env of chose to install nessary packages
<em>Note:</em> that the torch install command must be run separately as it redirects the index files for packages 

``pip install -I matplotlib scipy pandas numpy sklearn ipykernel ipython pathlib argparse datetime``
``pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`` see https://pytorch.org/get-started/locally/ for details if <i>cuda version is not 11.8</i> or you are not running on <i>linux</i> 
