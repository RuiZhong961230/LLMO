import re
import google.generativeai as genai
from copy import deepcopy


genai.configure(api_key="Your Gemini API")
model = genai.GenerativeModel("gemini-pro")


import numpy as np
import warnings
from ENAS_Data.robustness_dataset import RobustnessDataset
import os
warnings.filterwarnings("ignore")


Data = RobustnessDataset(path=os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "\\ENAS_Data")
DataName = "cifar10"
Metric = "clean"  # candicates = ["clean", "aa_apgd-ce@Linf", "aa_square@Linf", "fgsm@Linf", "pgd@Linf"]
Results = Data.query(
    data=[DataName],
    measure="accuracy",
    key=RobustnessDataset.keys_clean + RobustnessDataset.keys_adv + RobustnessDataset.keys_cc
)


def transfer(tmpIndi):
    uid = 0
    for i in range(len(tmpIndi)):
        uid += tmpIndi[i] * 5 ** i
    return int(uid)


def fit_func(indi):
    global Metric, Results, DataName
    uid = transfer(indi)
    if Metric == "clean":
        acc = Results[DataName][Metric]["accuracy"][Data.get_uid(uid)]
    else:
        acc = Results[DataName][Metric]["accuracy"][Data.get_uid(uid)][Data.meta["epsilons"][Metric].index(1.0)]
    return acc


def RunRand(func):
    global curFEs, curIter, MaxFEs, TrialRuns, DimSize, Metric
    All_Trial_Best = []
    for i in range(10):
        BestList = []
        curFEs = 1
        np.random.seed(2024 + 88 * i)
        BestIndi = [np.random.randint(0, 5), np.random.randint(0, 5), np.random.randint(0, 5), np.random.randint(0, 5), np.random.randint(0, 5), np.random.randint(0, 5)]
        BestFit = func(BestIndi)
        while curFEs < MaxFEs:

            CR = "Act as an optimizer for adversarial robustness neural architecture search. "
            Insight = "The objective of this task is to maximize the accuracy. "
            Statement = "There are 5 possible operations and 6 edges that need to be deployed. You need to specify a 6-bit array where the value in each index is within [0, 5). The current best solution is " + str(BestIndi) + " with the best accuracy " + str(BestFit) + ". "
            Experiment = "Give me one example in the array-like format."

            try:
                response = model.generate_content(CR + Insight + Statement + Experiment)
                numbers = re.findall(r'\d+', response.text)
                curIndi = [int(number) for number in numbers]
                for idx in range(len(curIndi)):
                    if curIndi[idx] > 4:
                        curIndi[idx] = 4
                    if curIndi[idx] < 0:
                        curIndi[idx] = 0

            except ValueError:
                curIndi = [np.random.randint(0, 5), np.random.randint(0, 5), np.random.randint(0, 5),
                           np.random.randint(0, 5), np.random.randint(0, 5), np.random.randint(0, 5)]
            curFit = func(curIndi)
            if curFit > BestFit:
                BestIndi = deepcopy(curIndi)
                BestFit = curFit
            curFEs += 1
            BestList.append(BestFit)
        All_Trial_Best.append(np.abs(BestList))
    np.savetxt("./LLMO_Data/ENAS/" + DataName + "_" + str(Metric) + ".csv", All_Trial_Best, delimiter=",")


def main(dataname):
    global MaxFEs, Results, DataName, Metric
    DataName = dataname
    MaxFEs = 3000

    Results = Data.query(
        data=[dataname],
        measure="accuracy",
        key=RobustnessDataset.keys_clean + RobustnessDataset.keys_adv + RobustnessDataset.keys_cc
    )
    Indicators = ["clean", "aa_apgd-ce@Linf", "aa_square@Linf", "fgsm@Linf", "pgd@Linf"]
    for i in range(len(Indicators)):
        Metric = Indicators[i]
        RunRand(fit_func)


if __name__ == "__main__":
    if os.path.exists('./LLMO_Data/ENAS') == False:
        os.makedirs('./LLMO_Data/ENAS')
    Datasets = ["cifar10", "cifar100"]
    for data in Datasets:
        main(data)


