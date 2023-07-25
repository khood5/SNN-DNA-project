# SNN-DNA-project
This is a project for Kent State Biology Dept. It is designed to read in Single-molecule experiment data and predicted if the experiment (if run for more time) would have a high/medium/low number of events
<p>

# Single-molecule experiment prediction model

This notebook is the step by step development process for making a SNN to predict the reactivity level and number of reactions in Single-molecule experiments
Data is provided via the Kent State University chemistry department. The model development is provided by the Kent State University computer science department

<table style="border:none;padding: 10px;margin: auto;">
    <tr style="border:none;padding: 10px;margin: auto;">
<td style="border:none;padding: 10px;margin: auto;">
    <b>Contacts from the chem dept.</b><br>
            Li Zuo <br>
            Email: lzuo4@kent.edu <br>
            Office: 303B Willems Hall (ISB) <br>
            WeChat: wxid_t2t99eeqkhih22 <br>
            <br>
            Dr. Hao Shen<br>
            Email: hshen7@kent.edu<br>
        </td>
        <td style="border:none;padding: 10px;margin: auto;">
            <b>Contacts from the CS dept.</b><br>
    Kendric Hood<br>
            Email: khood5@kent.edu<br>
            Office: 160 Math and Science Building (MSB)<br>
            WeChat: wxid_m66vu8fc422c12 <br>
            <br>
            Dr. Qiang Guan <br>
            Email: qguan@kent.edu<br>
        </td>
    </tr>
</table>
  
Use the following commands in your python env of choice to install nessary packages
<em>Note:</em> that the torch install command must be run separately as it redirects the index files for packages 

``pip install -I matplotlib scipy pandas numpy sklearn ipykernel ipython pathlib argparse datetime``
``pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`` see https://pytorch.org/get-started/locally/ for details if <i>cuda version is not 11.8</i> or you are not running on <i>linux</i> 
