import pandas as pd
import glob, os


def excel_to_df(load_src,save_src,opt_input):
    DataFramesArr =[]
    newDataArray=[]
    OPTnum=opt_input.split(",")
    load_path=load_src
    save_path=save_src+".xlsx"
    os.chdir(load_path)
    for file in glob.glob("*.xlsx"):
        DataFramesArr.append(pd.read_excel(file))
    if(opt_input):
        for index in range(len(OPTnum)):
            excl_merged=pd.concat(DataFramesArr, ignore_index=True)
            excl_merged.set_index('Src', inplace=True)
            excl_merged=excl_merged.loc[OPTnum[index]]
            excl_merged=excl_merged.sort_values(by="Time")
            newDataArray.append(excl_merged)
        summed_excel_merged=pd.concat(newDataArray)
    else:
        summed_excel_merged = pd.concat(DataFramesArr, ignore_index=True)
    summed_excel_merged.to_excel(r''+save_path, engine='xlsxwriter')


















