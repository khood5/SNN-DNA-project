import pandas as pd
from pathlib import Path  
import argparse
from argparse import RawTextHelpFormatter
import os
import csv

def convertExcelToSpikeTrain(file: str):
    settings = Settings()
    file_df = pd.read_excel(os.path.join(settings._INPUT_DIRECTORY,file),header=None,names=["Frame Number", "Intesity"])
    frames = []
    for l in range(0,len(file_df)-1):
        numberOfMissingFrames = int(file_df.iloc[l+1]["Frame Number"]) - int(file_df.iloc[l]["Frame Number"]) - 1 # -1 becouse two frames right after each other are 1 frame apart
        arrayOfZeros = numberOfMissingFrames * [0]
        currentSpikeEvent = [1] if settings._BINARY == True else [file_df.iloc[l]["Intesity"]] 
        frames += currentSpikeEvent + arrayOfZeros # missing frame are 0s plus the current frame
    frames += [1] if settings._BINARY == True else [file_df.iloc[-1]["Intesity"]] 
    return frames

# note: that depending on weather or not the spike train is binary or base on intesity the values in "spikeTrain" can be either 1/0 or double/0
# see binary argument for more details
def wrtieSpikeTrainToFile(spikeTrain: list, filename: str, indexFile: str):
    settings = Settings()
    outputSpikeTrainFile = os.path.join(f"{settings._OUTPUT_DIRECTORY}",filename)
    with open(outputSpikeTrainFile, 'w+', newline='') as outFile:
        write = csv.writer(outFile)
        for spike in spikeTrain : write.writerow ([spike])

    outputIndexFile = os.path.join(f"{settings._OUTPUT_DIRECTORY}",indexFile)
    totalNumberOfEvents = sum([1 for spike in spikeTrain if spike > 0])
    indexes.append([outputSpikeTrainFile,getClassOfSpikeTrain(totalNumberOfEvents)])
    with open(outputIndexFile, 'a+', newline='') as outFile:
        write = csv.writer(outFile)
        write.writerows(indexes)

def getClassOfSpikeTrain(spikeCount: int):
    settings = Settings()
    if settings._SETTINGS_FILE == None:
        return spikeCount
    
    high = -1
    low = -1
    with open(settings._SETTINGS_FILE, newline='') as settings:
        thresholdReader = csv.reader(settings, delimiter=',')
        for row in thresholdReader:
            if row[0].lower().strip() == "high":
                high = int(row[1])
            elif row[0].lower().strip() == "low":
                low = int(row[1])
    if spikeCount >= high:
        return "high"
    if spikeCount <= low:
        return "low"
    return "medium"
        

class Settings:
    _INPUT_DIRECTORY = None     # directory with the files to be converted to spike trains
    _OUTPUT_DIRECTORY = None    # output directory for spike train files     
    _BINARY = None              # if true then events are converted to 1 or spike 0 for no spike, otherwise intensity of event is used for spike
    _SETTINGS_FILE = None       # file path to seetings; settings should define the high and low thresholds for classes (see settings argument for more details)

    def __new__(cls):
            if not hasattr(cls, 'instance'):
                    cls.instance = super(Settings, cls).__new__(cls)
            return cls.instance 

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='''Process Single-molecule experiments data to add frames with 0 reactions and convert from excel to csv. \n\
    Creates index file that has the file path of each csv file (relative to where this program runs) \n\
    and that files class. Each file is one spike train used for training. \n''', 
                                    formatter_class=RawTextHelpFormatter )
    parser.add_argument('input_directory', help='Directory with input files\n',
                        type=str)
    parser.add_argument('output_directory', help='Directory for output files\n',
                        type=str)
    parser.add_argument('--binary', help="switches to binary spikes events default is spikes events based on reaction intesity (floating point value)\n", 
                        action=argparse.BooleanOptionalAction)
    parser.add_argument('--settings', help='''Settings file: classes are based on the sum of total events, thier intesity is not considerd when counting events \n\
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


    args = parser.parse_args()
    settings = Settings()
    settings._INPUT_DIRECTORY = args.input_directory
    settings._OUTPUT_DIRECTORY = args.output_directory
    settings._BINARY = args.binary
    settings._SETTINGS_FILE = args.settings
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
        spikeTrain = convertExcelToSpikeTrain(f)
        wrtieSpikeTrainToFile(spikeTrain, f"spikeTrain_{fileNameNumber}.csv", "index.csv")
        fileNameNumber += 1
        
        
    
        