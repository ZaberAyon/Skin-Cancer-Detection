import pandas as pd


def dataset_info_from_csv(filepath):
    dicti = {}
    dataFrame = pd.read_csv(filepath)
    filenames = dataFrame['filename'].to_list()
    diag = dataFrame['diagnoses'].to_list()

    for i in range(len(filenames)):
        dicti[filenames[i]] = diag[i]

    return dicti

res = dataset_info_from_csv("D:\Projects\Skin_cancer\Skin-Cancer-Detection\minimized.csv")
print(res)

def check_acc_cv(filepath, dicti):
    total_curr = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    total_melanoma = 0
    df = pd.read_csv(filepath)
    filenames = df['filename'].to_list()
    result = df['result'].to_list()
    total = len(filenames)
    for i in range(total):
        res = result[i]
        fname = filenames[i].split(".")[0]
        diag = dicti[fname]
        if(diag == res):
            total_curr = total_curr+1
            if(diag == 'melanoma'):
                total_melanoma = total_melanoma +1
        if(diag == 'benign' and res == 'benign'):
            tp = tp+1
        if(diag == 'melanoma' and res == 'melanoma'):
            tn = tn+1
        if(diag == 'melanoma' and res == 'benign'):
            fp = fp+1
        if(diag == 'benign' and res == 'melanoma'):
            fn = fn+1
    return total_curr, total_melanoma, tp, fp, tn, fn, total


total_curr, total_melanoma, tp, fp, tn, fn, total = check_acc_cv("D:\Projects\Skin_cancer\Skin-Cancer-Detection\CVoutput.csv", res)

acc = (total_curr/total)* 100
print("Accuracy of TDS: "+str(acc)+"%")

print("Total Melanoma = ",total_melanoma)
print("True Positive: ",tp)
print("True Negetive", tn)
print("False Positive", fp)
print("False Negetive", fn)


def create_half_dataset(filepath, dicti):
    df = pd.read_csv(filepath)
    filenames = df['filename'].to_list()
    result = []
    a_minor = df['a_minor'].to_list()
    a_major = df['a_major'].to_list()
    borderirr = df['borderirr'].to_list()
    colorind = df['colorind'].to_list()
    diam = df['diameter'].to_list()
    total = len(filenames)
    for i in range(total):
        fname = filenames[i].split(".")[0]
        diag = dicti[fname]
        result.append(diag)
    resultDf = pd.DataFrame(list(zip(filenames,a_minor,a_major,borderirr,colorind,diam,result)),columns=['filename','a_minor','a_major','borderirr','colorind','diameter','result'])
    resultDf.to_csv("polished.csv")

create_half_dataset("D:\Projects\Skin_cancer\Skin-Cancer-Detection\CVoutput.csv", res)
