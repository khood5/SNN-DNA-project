import pandas as pd
from pathlib import Path  
import argparse
from argparse import RawTextHelpFormatter
from datetime import datetime
import os
import csv
import hashlib

def convertExcelToSpikeTrain(file: str):
    settings = Settings()
    file_df = pd.read_excel(os.path.join(settings._INPUT_DIRECTORY,file),header=None,names=["Frame Number", "Intesity"])
    file_df = shorten(file_df)
    frames = []
    lastFrameNumber = 0
    for l in range(0,len(file_df)):
        numberOfMissingFrames = int(file_df.iloc[l]["Frame Number"]) - lastFrameNumber - 1 # -1 becouse two frames right after each other are 1 frame apart
        lastFrameNumber = int(file_df.iloc[l]["Frame Number"])
        arrayOfZeros = numberOfMissingFrames * [0]
        currentSpikeEvent = [1] if settings._BINARY == True else [file_df.iloc[l]["Intesity"]] 
        frames += arrayOfZeros + currentSpikeEvent # missing frame are 0s plus the current frame
    
    missingEventsFromEndOfExperment = int((settings._LENGTH * settings._FPS) - len(frames))
    frames += [0] * missingEventsFromEndOfExperment
    return frames

def shorten(file_df: pd.DataFrame):
    settings = Settings()
    cutoffIndex = 1
    lastFrame = file_df.iloc[0]["Frame Number"]
    while (lastFrame/settings._FPS) < settings._LENGTH - 1 and cutoffIndex < len(file_df):
        lastFrame = file_df.iloc[cutoffIndex]["Frame Number"]
        cutoffIndex += 1
    file_df = file_df.drop(list(range(cutoffIndex -1, len(file_df))))
    return file_df

# note: that depending on weather or not the spike train is binary or base on intesity the values in "spikeTrain" can be either 1/0 or double/0
# see binary argument for more details
def wrtieSpikeTrainToFile(spikeTrain: list, SpikeTrainFilename: str, sourceFilename: str, indexes: list):
    settings = Settings()
    outputSpikeTrainFile = os.path.join(f"{settings._OUTPUT_DIRECTORY}",SpikeTrainFilename)
    with open(outputSpikeTrainFile, 'w+', newline='') as outFile:
        write = csv.writer(outFile)
        for spike in spikeTrain : write.writerow ([spike])
    totalNumberOfEvents = len(pd.read_excel(os.path.join(settings._INPUT_DIRECTORY,sourceFilename),header=None,names=["Frame Number", "Intesity"]))
    
    spikeTrainClass = getClassOfSpikeTrain(totalNumberOfEvents)
    indexes.append([SpikeTrainFilename,spikeTrainClass])


def getClassOfSpikeTrain(spikeCount: int):
    settings = Settings()
    
    if settings._SETTINGS_FILE == None:
        return spikeCount
    
    high = -1
    low = -1
    with open(settings._SETTINGS_FILE, newline='') as thresholdSettings:
        thresholdReader = csv.reader(thresholdSettings, delimiter=',')
        for row in thresholdReader:
            if row[0].lower().strip() == "high":
                high = int(row[1])
            elif row[0].lower().strip() == "low":
                low = int(row[1])
    
    if spikeCount >= high:
        return settings._HIGH_CLASS_CODE
    if spikeCount <= low:
        return settings._LOW_CLASS_CODE
    return settings._MEDIUM_CLASS_CODE
        

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
    _NUM_CLASS          = None  # if true the acutal count is used for the class (i.e. 108 instead of high)
    _SILENT             = None  # if true program should not output anything

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
    parser.add_argument('-n','--num_class', help="the acutal count is used for the class (i.e. 108 instead of high)\n", 
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
    parser.add_argument('-0','--silent', help="prevents any output\n", 
                        action=argparse.BooleanOptionalAction)


    args = parser.parse_args()
    settings = Settings()
    ### user defined settings ###
    settings._INPUT_DIRECTORY = args.input_directory
    settings._OUTPUT_DIRECTORY = args.output_directory
    settings._BINARY = args.binary
    settings._NUM_CLASS = args.num_class
    settings._SETTINGS_FILE = args.settings
    settings._LENGTH = args.length
    settings._FPS = args.frams_per_second
    settings._SILENT = args.silent
    ### Fixed settings ###
    settings._HIGH_CLASS_CODE = HIGH_CLASS_CODE
    settings._MEDIUM_CLASS_CODE = MEDIUM_CLASS_CODE
    settings._LOW_CLASS_CODE = LOW_CLASS_CODE
    
    if settings._SILENT != True:
        print(f"reading files from  :{Path(settings._INPUT_DIRECTORY)}")
        print(f"creating files in   :{Path(settings._OUTPUT_DIRECTORY)}")
        if settings._SETTINGS_FILE:
            print(f"settings file       :{Path(settings._SETTINGS_FILE)}")
        else:
            print(f"spike count")
    
    if os.path.isdir(settings._INPUT_DIRECTORY):
        inputFiles = os.listdir(settings._INPUT_DIRECTORY)
    elif os.path.isfile(settings._INPUT_DIRECTORY):
        inputFiles = [os.path.basename(settings._INPUT_DIRECTORY)]
        settings._INPUT_DIRECTORY = Path(settings._INPUT_DIRECTORY).parent.absolute()
    else:
        print(f"ERROR: {settings._INPUT_DIRECTORY} is not a file or directory")
        
    indexes = []
    for f in inputFiles:
        supported_file_types = ["xls", "xlsx", "xlsm", "xlsb", "xlt", "xls", "xml", "xlw", "xlr"]
        if f.split(".")[-1] not in supported_file_types:
            continue # skip none excel files
        spikeTrain = convertExcelToSpikeTrain(f)
        h = hashlib.sha256()
        h.update((f+str(datetime.now().timestamp())).encode('utf-8'))
        wrtieSpikeTrainToFile(spikeTrain, f"{h.hexdigest()}.csv", f, indexes)
    outputIndexFile = os.path.join(f"{settings._OUTPUT_DIRECTORY}",INDEX_FILE_NAME)
    with open(outputIndexFile, 'a+', newline='') as outFile:
        write = csv.writer(outFile)
        write.writerows(indexes)
    settings.save_settings()
    
        
        
    
        