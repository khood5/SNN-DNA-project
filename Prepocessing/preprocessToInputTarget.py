import pandas as pd
from pathlib import Path  
import argparse
from argparse import RawTextHelpFormatter
from datetime import datetime
import os
import csv
import copy
import warnings
import hashlib
warnings.simplefilter(action='ignore', category=FutureWarning)

def convertExcelToSpikeTrain(file: str):
    settings = Settings()
    file_df = pd.read_excel(os.path.join(settings._INPUT_DIRECTORY,file),header=None,names=["Frame Number", "Intesity"])
    targetDf = copy.deepcopy(file_df)
    inputDf = shorten(copy.deepcopy(file_df), settings._LENGTH)
    if(settings._REPLAY):
        inputDf = replay(copy.deepcopy(inputDf),targetDf[:]["Frame Number"].max())
    
    min = float(targetDf[:]["Intesity"].min()) # max/min need to be the same for both target and input
    max = float(targetDf[:]["Intesity"].max())
    
    input = []
    for l in range(0,len(inputDf)):
        currentSpikeEvent = 1 
        if settings._BINARY != True:
            currentSpikeEvent = (float(inputDf.iloc[l]["Intesity"]) - min)/(max-min) * settings._Y_SCALE
            currentSpikeEvent = int(currentSpikeEvent)
            for i in range(currentSpikeEvent):
                input.append([i,(inputDf.iloc[l]["Frame Number"]/settings._FPS)])
        else:
             input.append([currentSpikeEvent,(inputDf.iloc[l]["Frame Number"]/settings._FPS)])
    target = []
    for l in range(0,len(targetDf)):
        currentSpikeEvent = 1 
        if settings._BINARY != True:
            currentSpikeEvent = (float(targetDf.iloc[l]["Intesity"]) - min)/(max-min) * settings._Y_SCALE
            currentSpikeEvent = int(currentSpikeEvent)
            for i in range(currentSpikeEvent):
                target.append([i,(targetDf.iloc[l]["Frame Number"]/settings._FPS)])
        else:
             target.append([currentSpikeEvent,(targetDf.iloc[l]["Frame Number"]/settings._FPS)])
    return (input, target)

def shorten(file_df: pd.DataFrame, length: int):
    settings = Settings()
    cutoffIndex = 1
    lastFrame = file_df.iloc[0]["Frame Number"]
    while lastFrame*(1/settings._FPS) < length - 1 and cutoffIndex < len(file_df):
        lastFrame = file_df.iloc[cutoffIndex]["Frame Number"]
        cutoffIndex += 1
    file_df = file_df.drop(list(range(cutoffIndex -1, len(file_df))))
    return file_df

def replay(file_df: pd.DataFrame, targetLength: int):
    replayedCopy = file_df.copy(deep=True)
    max = int(file_df[:]["Frame Number"].max())
    while max < targetLength:
        file_df["Frame Number"] = file_df["Frame Number"] + max
        replayedCopy = pd.concat([replayedCopy,file_df], ignore_index=True, axis=0)
        max = int(file_df[:]["Frame Number"].max())
    return replayedCopy

# note: that depending on weather or not the spike train is binary or base on intesity the values in "spikeTrain" can be either 1/0 or double/0
# see binary argument for more details
def wrtieSpikeTrainToFile(spikeTrain: list, SpikeTrainFilename: str):
    settings = Settings()
    
    outputSpikeTrainFile = os.path.join(f"{settings._OUTPUT_DIRECTORY}",SpikeTrainFilename)
    with open(outputSpikeTrainFile, 'w+', newline='') as outFile:
        write = csv.writer(outFile)
        for spike in spikeTrain : write.writerow (spike)
        

class Settings:
    _INPUT_DIRECTORY    = None  # directory with the files to be converted to spike trains
    _OUTPUT_DIRECTORY   = None  # output directory for spike train files     
    _BINARY             = None  # if true then events are converted to 1 or spike 0 for no spike, otherwise intensity of event is used for spike
    _SETTINGS_FILE      = None  # file path to seetings; settings should define the high and low thresholds for classes (see settings argument for more details)
    _LENGTH             = None  # number of seconds to use for training (events after this time will not be included in the spike trains)
    _FPS                = None  # frames per second 
    _EVENTS_PER_SEC     = None  # number of events per second of experment 
    _HIGH_CLASS_CODE    = None  # High class numerical value
    _MEDIUM_CLASS_CODE  = None  # Medium class numerical value
    _LOW_CLASS_CODE     = None  # Low class numerical value
    _Y_SCALE            = None  # Y scale for non-binary event conversion
    _REPLAY             = None  # Controls weather shorten spiketrain is repeated to match the length of the original 

    def __new__(cls):
            if not hasattr(cls, 'instance'):
                    cls.instance = super(Settings, cls).__new__(cls)
            return cls.instance 
    def save_settings(self):
                dt_string = datetime.now().strftime("%d.%m.%Y_%H-%M-%S-%f")
                try:
                    f = open(f"./preprocess_logs/preprocess_{dt_string}_saved_args.csv", "w+")
                except FileNotFoundError as e: 
                    print("!!! Missing 'preprocess_logs' directory\n writing logs to this directory")
                    f = open(f"./preprocess_{dt_string}_saved_args.csv", "w+")
            
                f.write("_INPUT_DIRECTORY,_OUTPUT_DIRECTORY,_BINARY,_SETTINGS_FILE,_LENGTH,_FPS\n")
                f.write(f"{self._INPUT_DIRECTORY},{self._OUTPUT_DIRECTORY},{self._BINARY},{self._SETTINGS_FILE},{self._LENGTH},{self._FPS}\n")
                f.write(f"run at {dt_string}")
                f.close()
        
if __name__ == "__main__":
    DEFAULT_FPS = 20
    DEFAULT_LENGTH = 5 * 60
    INDEX_FILE_NAME = "index.csv"
    HIGH_CLASS_CODE = 2
    MEDIUM_CLASS_CODE = 1
    LOW_CLASS_CODE = 0
    Y_SCALE = 200
    parser = argparse.ArgumentParser(description='''Process Single-molecule experiments data to add frames with 0 reactions and convert from excel to csv. \n\
    Creates index file that has the file path of each csv file (relative to where this program runs) \n\
    and that files class. Each file is one spike train used for training. \n''', 
                                    formatter_class=RawTextHelpFormatter )
    parser.add_argument('input_directory', help='Directory with input files\n',
                        type=str)
    parser.add_argument('output_directory', help='Directory for output files\n',
                        type=str)
    parser.add_argument('-b','--binary', help="switches to binary spikes events default is spikes events based on reaction intesity (floating point value)\n", 
                        action=argparse.BooleanOptionalAction)
    parser.add_argument('-r','--replay', help="switchs to repeating the input spike train to match the target length\n", 
                        action=argparse.BooleanOptionalAction)
    parser.add_argument('-s','--settings', help='''Settings file: classes are based on the sum of total events, thier intesity is not considerd when counting events \n\
    high: number of recoreded events to be considerd in the high class \n\
    low:  number of recoreded events to be considerd in the low class \n\
    for values between high and low they are considerd medium \n\
    example: \n\
    high, 300 \n\
    low, 230 \n\
    \n\
    If no settings file is supplied (i.e. None) then raw event counts will be used for classes \n\
    ''',
    type=str)
    parser.add_argument('-l','--length', help='''number of seconds to use for training (events after this time will not be included in the spike trains)\n\
                        DEFAULT: {}'''.format(DEFAULT_LENGTH), 
                        type=float,
                        default=DEFAULT_LENGTH)
    parser.add_argument('-fps','--frams_per_second', help='''number of frames per second \n \
                        DEFAULT: {}'''.format(DEFAULT_FPS), 
                        type=int,
                        default=DEFAULT_FPS)
    parser.add_argument('-y','--y_scale', help='''if non-binary events then this is the scale that the y value will be between (0-y_scale) \n \
                        DEFAULT: {}'''.format(Y_SCALE), 
                        type=int,
                        default=Y_SCALE)


    args = parser.parse_args()
    settings = Settings()
    ### user defined settings ###
    settings._INPUT_DIRECTORY = args.input_directory
    settings._OUTPUT_DIRECTORY = args.output_directory
    settings._BINARY = args.binary
    settings._REPLAY = args.replay
    settings._SETTINGS_FILE = args.settings
    settings._LENGTH = args.length
    settings._FPS = args.frams_per_second
    settings._Y_SCALE = args.y_scale
    ### Fixed settings ###
    settings._HIGH_CLASS_CODE = HIGH_CLASS_CODE
    settings._MEDIUM_CLASS_CODE = MEDIUM_CLASS_CODE
    settings._LOW_CLASS_CODE = LOW_CLASS_CODE
    
    print(f"reading files from  :{Path(settings._INPUT_DIRECTORY)}")
    print(f"creating files in   :{Path(settings._OUTPUT_DIRECTORY)}")
    print(f"settings file       :{Path(settings._SETTINGS_FILE)}")
    
    inputFiles = os.listdir(args.input_directory)
    indexes = []
    fileNameNumber = 1
    for f in inputFiles:
        supported_file_types = ["xls", "xlsx", "xlsm", "xlsb", "xlt", "xls", "xml", "xlw", "xlr"]
        if f.split(".")[-1] not in supported_file_types:
            continue # skip none excel files
        input, target = convertExcelToSpikeTrain(f)
        
        SpikeTrainFilename = f"spikeTrain_{fileNameNumber}.csv"
        targetSpikeTrainFilename = f"spikeTrain_{fileNameNumber}_target.csv"
        h = hashlib.sha256()
        h.update(SpikeTrainFilename.encode('utf-8'))
        SpikeTrainFilename = h.hexdigest()
        h.update(targetSpikeTrainFilename.encode('utf-8'))
        targetSpikeTrainFilename = h.hexdigest()
        
        wrtieSpikeTrainToFile(input, SpikeTrainFilename)
        wrtieSpikeTrainToFile(target, targetSpikeTrainFilename)
        indexes.append([SpikeTrainFilename,targetSpikeTrainFilename])
        fileNameNumber += 1
    outputIndexFile = os.path.join(f"{settings._OUTPUT_DIRECTORY}",INDEX_FILE_NAME)
    with open(outputIndexFile, 'w+', newline='') as outFile:
        write = csv.writer(outFile)
        write.writerows(indexes)
    
    settings.save_settings()
    
        
        
    
        