'''
2022.10.11
licheng
idea：垂直相界搜索宽温域超弹性材料
如果一个合金体系，某一成分稍微变动一下（一般认为是1%），相变温度发生剧烈变动
相变温度变动为200以上到-50以下即为剧烈变动，则合金体系必须含有Hf和Zr，变动Ni及Ni的替代元素来降低相变温度
'''
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.model_selection import cross_validate,LeaveOneOut,cross_val_predict
import matplotlib.pyplot as plt
import csv
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import copy
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.model_selection import RandomizedSearchCV
import scienceplots
# 获取斜率特征，slope为不同代的元素斜率特征集合，data为只含原子百分比的信息集合，m为第几代斜率
def SlopeGet(Slope,m,data):
    print("正在生成斜率特征")
    slope = Slope.iloc[m].values.tolist()
    if m >= 12 and m <= 15:
        # TiNum = data.iloc[:, 0] + data.iloc[:, 9] + data.iloc[:, 10] + data.iloc[:, 8]/2
        # NiNum = data.iloc[:, 1] + data.iloc[:, 2] + data.iloc[:, 3] + data.iloc[:, 4] + data.iloc[:, 5] + data.iloc[:, 6] + data.iloc[:, 7] + data.iloc[:, 8]/2
        TiNum = data.iloc[:, 0] + data.iloc[:, 9] + data.iloc[:, 10]
        NiNum = data.iloc[:, 1] + data.iloc[:, 2] + data.iloc[:, 3] + data.iloc[:, 4] + data.iloc[:, 5] + data.iloc[:, 6] + data.iloc[:, 7]
        # TNNum = TiNum+NiNum
        # TiNum = TiNum/TNNum*100
        # NiNum = NiNum/TNNum*100
        data1 = copy.deepcopy(data)
        TiNum = np.array(TiNum)
        NiNum = np.array(NiNum)
        TiNUM = np.maximum((TiNum - 50), 0)
        NiNUM = np.maximum((NiNum - 50), 0)
        # TiNUM = np.maximum((TiNum/(TiNum + NiNum))*100-50, 0)
        # NiNUM = np.maximum((NiNum/(TiNum + NiNum))*100-50, 0)
        data1.iloc[:, 0] = TiNUM
        data1.iloc[:, 1] = NiNUM
        new_slope1 = data1.dot(slope)
        new_slope1 = new_slope1 / 100
    else:
        new_slope1 = data.dot(slope)
        new_slope1 = new_slope1 / 100
    return new_slope1

#特征数据集中添加数据,data_Feature为原子信息特征集合，data为只含原子百分比的信息集合
def featureSet(data_Feature,data,m):
    print('正在生成非斜率特征')
    OldFeature = ['mr','ar_c','en','anum','ven','ea','dor','mass','volume','energy1','CE','EBE','YM','cs','Tm']
    data2 = pd.DataFrame()#用来存放虚拟成分的特征集合
    for i in range(data_Feature.shape[0]):
        feature = data_Feature.iloc[i,:].values.tolist()
        #feature.append(0)
        Feature = data.dot(feature)
        #部分表格的数据为百分比数据，需除100
        Feature = Feature / m
        data2.insert(i, OldFeature[i], Feature)
    # 增加价电子浓度特征
    cv = data2['ven'] / data2['anum']
    data2.insert(data2.shape[1], 'cv', cv)
    return data2
# data1为元素的除斜率外特征集合,dataframe变量；data2为dataframe变量,为待预测的成分集合；data3为dataframe变量,每一代更新的斜率特征；
# m1为修正系数,默认为100；m2选择的第几代斜率，默认为14
def getPredictDataFeature(data1,data2,data3,m1=100,m2=14):
    data_space_feature = featureSet(data_Feature=data1,data=data2,m=m1)
    newslope = SlopeGet(Slope=data3,m=m2,data=data2)
    data_space_feature.insert(data_space_feature.shape[1],'slope_imp',newslope)
    return data_space_feature

# data1为dataframe变量，为待搜索的成分组合；composite为变动的的元素名；crange为整数变量，为相应的变动范围，例如0到11；
def SearchLargeSETempRange(data1,Vary):
    # 采用穷举法，搜索只有两个成分变动1的合金体系，比较hp的变化，如果>=300,则记录对应的合金成分编号，记录hp的差值
    dataR = pd.DataFrame(columns=['alloy1','alloy2','hpVary'])
    for i in tqdm(range(data1.shape[0])):
        loclist = []
        dataVary = data1.loc[i]-data1
        # 针对dataVary，需要统计的量为行最大值，行零值个数，hp/sum
        dataVary1 = dataVary.drop(['hp'],axis=1)
        listMax = list(dataVary1.max(axis=1))
        dataVary2 = dataVary.replace(0,np.nan)
        listNum0 = dataVary2.isnull().sum(axis=1)
        listHpVary = abs(dataVary.loc[:,'hp']/dataVary1.max(axis=1))
        dataVary.insert(dataVary.shape[1],'listNum0',listNum0)
        dataVary.insert(dataVary.shape[1], 'listMax', listMax)
        dataVary.insert(dataVary.shape[1], 'listHpVary', listHpVary)
        dataVaryWant = dataVary[(dataVary['listNum0']==9)&(dataVary['listHpVary']>=Vary)&(dataVary['listMax']==1)].copy()
        if not dataVaryWant.empty:
            listalloy1 = [m*0+i for m in range(dataVaryWant.shape[0])]
            dataRR = pd.DataFrame()
            dataRR.insert(0, 'alloy1', listalloy1)
            dataRR.insert(1, 'alloy2', dataVaryWant.index.values)
            dataRR.insert(2, 'hpVary', list(dataVaryWant.loc[:,'listHpVary']))
            dataR = pd.concat([dataR,dataRR])
    dataR = dataR.sort_values(by='hpVary')
    return dataR

def GetModel(data,feature,):
    X = data.loc[:, feature[0]]
    X_scaler = StandardScaler().fit_transform(X)
    y = data.loc[:, feature[1]]
    params_poly = {'kernel': ['poly'], 'alpha': [i for i in np.arange(0.01, 10, 0.1)],
                   'gamma': [i for i in np.arange(0.1, 2, 0.1)], 'coef0': [0.01, 0.1, 1, 100],
                   'degree': [1, 2, 3]}
    bestscore = -10000
    for m in range(10):
        krr_poly = RandomizedSearchCV(KernelRidge(), params_poly, cv=10, verbose=0,
                                      scoring='neg_mean_squared_error', n_iter=30,
                                      random_state=m)  # cv=10即10折交叉验证
        krr_poly.fit(X_scaler,y)  # 对给定数据集选取最佳参数
        if krr_poly.best_score_ > bestscore:
            bestscore = krr_poly.best_score_
            model = krr_poly.best_estimator_
    return model

# 得到虚拟空间的预测目标数据,model为模型的具体超参数,feature为list存储了目标性能和对应的特征组合
# data_train为已经得到的训练数据,data_predict为待预测的成分集合,data_predict_feature为为待预测的成分特征集合
def GetPredictData(model,feature,data_train,data_predict,data_predict_feature):
    print('正在预测%s性能'%feature[1])
    x = data_train.loc[:, feature[0]]
    xx = data_predict_feature.loc[:, feature[0]]
    scaler = StandardScaler()
    scaler.fit(x)
    x_scaler = scaler.transform(x)
    y = data_train.loc[:, feature[1]]
    reg_hp = model.fit(x_scaler, y)
    xx_scaler = scaler.transform(xx)
    predict_feature = reg_hp.predict(xx_scaler)
    # y = data_train.loc[:, feature[1]]
    # reg_hp = model.fit(x, y)
    # predict_feature = reg_hp.predict(xx)
    data_predict1 = copy.deepcopy(data_predict)
    data_predict1.insert(data_predict1.shape[1],feature[1],predict_feature)
    return data_predict1

# 得到虚拟空间的预测目标数据,model为模型的具体超参数,feature为list存储了目标性能和对应的特征组合
# data_train为已经得到的训练数据,data_predict为待预测的成分集合,data_predict_feature为为待预测的成分特征集合
def GetPredictData1(model,feature,data_train,data_predict,data_predict_feature):
    # 不对特征做归一化
    print('正在预测%s性能'%feature[1])
    x = data_train.loc[:, feature[0]]
    xx = data_predict_feature.loc[:, feature[0]]
    y = data_train.loc[:, feature[1]]
    reg_hp = model.fit(x, y)
    predict_feature = reg_hp.predict(xx)
    data_predict1 = copy.deepcopy(data_predict)
    data_predict1.insert(data_predict1.shape[1],feature[1],predict_feature)
    return data_predict1


def GetDataMeanAndStd(data_predict,data_predict_feature,data_train,feature,model):
    # 此函数获取虚拟空间预测的均值和标准差，data_predict为待预测的虚拟空间,data_predict_feature为待预测的虚拟空间特征集合,
    # data_train为准备好的训练数据,
    # feature为list，有两个元素，1个为性能的特征组合，1个为目标性能,model为预测模型,mul为修正系数，hp为1，enthalpy和hysteresis为-1
    print('正在预测%s性能的均值和标准差' % feature[1])
    fitted = data_predict.iloc[:,[-1]].copy()
    data_predict1 = data_predict.drop(labels=feature[1], axis=1)
    for i in tqdm(range(1, 1001)):
        data_boot = data_train.sample(data_train.shape[1], replace=True, axis=0, random_state=i + 1)  # 有放回自助抽样1000次行
        data_boot_feature = GetPredictData(model,feature,data_boot,data_predict1,data_predict_feature)
        data_boot_krr_fitted = data_boot_feature.iloc[:,-1]
        fitted.insert(i, 'pred%d' % i, data_boot_krr_fitted)
    # fitted_std = fitted.std(axis=1)**0.5
    # data_predict.insert(data_predict.shape[1], 'mean', fitted.mean(axis=1))
    data_predict.insert(data_predict.shape[1], 'std', fitted.std(axis=1)**0.5)
    return data_predict
# path为文件路径, order为需求的文件列名排序, targetorder为需要的成分组合排序,target为目标性能
def GetHavedData(path,order,targetorder,target):
    data= pd.read_csv(filepath_or_buffer=path)
    order1 = copy.deepcopy(order)
    order1.append(target)
    if path == 'C:\\licheng\\slope-fitting\\hp\\SMA.data.training2.csv':
        data = pd.read_csv(filepath_or_buffer=path, usecols=[i for i in range(1, 13)])
        order1 = copy.deepcopy(order)
        order1.append(target)
        data.columns = order1
        # 删除hp中的TiPdCr成分
        data = data.loc[data['Ni'] != 0].copy()
        data = data.reset_index(drop=True)
    else:
        data = data[order1]
    targetpass = [y for y in order if y not in targetorder]
    for i in targetpass:
        data = data[data[i]==0].copy()
    return data
# 将元素信息转化为字符串信息
# list1为输入的元素含量信息
def DataToStr(list1):
    order = ['Ti', 'Ni', 'Cu', 'Fe', 'Pd', 'Co', 'Mn', 'Cr', 'Nb', 'Zr', 'Hf']
    strWanted = ''
    for i in range(len(order)):
        if i < 2 :
            elementNumber = round(list1[i],1)
        else:
            elementNumber = int(list1[i])
        if elementNumber != 0 :
            strWanted = strWanted + order[i] + str(elementNumber)
    return strWanted
# 根据SearchLargeSETempRange()的结果组合产生更加直观的对比结果
# data1为SearchLargeSETempRange()产生的结果,data2为SearchLargeSETempRange()计算的初始对应数据
def SearchLargeSETempRange1(data1,data2):
    alloy1, hp1, alloy2, hp2 = [], [], [], []
    for i in tqdm(range(data1.shape[0])):
        alloyNumber1 = data1.iloc[i,0]
        list1 = data2.loc[alloyNumber1].values
        str1 = DataToStr(list1)
        alloyNumber2 = data1.iloc[i,1]
        list2 = data2.loc[alloyNumber2].values
        str2 = DataToStr(list2)
        if list1[-1] > list2[-1]:
            alloy1.append(str1)
            hp1.append(list1[-1])
            alloy2.append(str2)
            hp2.append(list2[-1])
        else:
            alloy1.append(str2)
            hp1.append(list2[-1])
            alloy2.append(str1)
            hp2.append(list1[-1])
    dataRR = pd.DataFrame()
    dataRR.insert(0, 'alloy1', alloy1)
    dataRR.insert(1, 'hp1', hp1)
    dataRR.insert(2, 'alloy2', alloy2)
    dataRR.insert(3, 'hp2', hp2)
    dataRR.insert(4, 'hpVary', data1.loc[:,'hpVary'].values)
    return dataRR

def SearchLargeSETempRange2(data1,data2):
    # 根据SearchLargeSETempRange()的结果组合产生更加直观的对比结果
    # data1为SearchLargeSETempRange()产生的结果,data2为SearchLargeSETempRange()计算的初始对应数据
    alloy1, hp1, std1, alloy2, hp2, std2 = [], [], [], [], [], []
    for i in tqdm(range(data1.shape[0])):
        alloyNumber1 = data1.iloc[i,0]
        list1 = data2.loc[alloyNumber1].values
        str1 = DataToStr(list1)
        alloyNumber2 = data1.iloc[i,1]
        list2 = data2.loc[alloyNumber2].values
        str2 = DataToStr(list2)
        if list1[-1] > list2[-1]:
            alloy1.append(str1)
            hp1.append(list1[-2])
            std1.append(list1[-1])
            alloy2.append(str2)
            hp2.append(list2[-2])
            std2.append(list2[-1])
        else:
            alloy1.append(str2)
            hp1.append(list2[-2])
            std1.append(list2[-1])
            alloy2.append(str1)
            hp2.append(list1[-2])
            std2.append(list1[-1])
    dataRR = pd.DataFrame()
    dataRR.insert(0, 'alloy1', alloy1)
    dataRR.insert(1, 'hp1', hp1)
    dataRR.insert(2, 'std1', std1)
    dataRR.insert(3, 'alloy2', alloy2)
    dataRR.insert(4, 'hp2', hp2)
    dataRR.insert(5, 'std2', std2)
    dataRR.insert(6, 'hpVary', data1.loc[:,'hpVary'].values)
    Max_std = np.sum([std1, std2], axis=0).tolist()
    dataRR.insert(7, 'Max_std', Max_std)
    return dataRR
# data1为dataframe变量，为待搜索的成分组合；composite为变动的的元素名；crange为整数变量，为相应的变动范围，例如0到11；
def SearchLargeSETempRange3(data1):
    # 采用穷举法，搜索只有两个成分变动1的合金体系，比较hp的变化，如果<=50,则记录对应的合金成分编号，记录hp的差值
    dataR = pd.DataFrame(columns=['alloy1','alloy2','hpVary'])
    for i in tqdm(range(data1.shape[0])):
        loclist = []
        dataVary = data1.loc[i]-data1
        # 针对dataVary，需要统计的量为行最大值，行零值个数，hp/sum
        dataVary1 = dataVary.drop(['hp'],axis=1)
        listMax = list(dataVary1.max(axis=1))
        dataVary2 = dataVary.replace(0,np.nan)
        listNum0 = dataVary2.isnull().sum(axis=1)
        listHpVary = abs(dataVary.loc[:,'hp']/dataVary1.max(axis=1))
        dataVary.insert(dataVary.shape[1],'listNum0',listNum0)
        dataVary.insert(dataVary.shape[1], 'listMax', listMax)
        dataVary.insert(dataVary.shape[1], 'listHpVary', listHpVary)
        dataVaryWant = dataVary[(dataVary['listNum0']==9)&(dataVary['listHpVary']<=50)&(dataVary['listMax']==1)].copy()
        if not dataVaryWant.empty:
            listalloy1 = [m*0+i for m in range(dataVaryWant.shape[0])]
            dataRR = pd.DataFrame()
            dataRR.insert(0, 'alloy1', listalloy1)
            dataRR.insert(1, 'alloy2', dataVaryWant.index.values)
            dataRR.insert(2, 'hpVary', list(dataVaryWant.loc[:,'listHpVary']))
            dataR = pd.concat([dataR,dataRR])
    dataR = dataR.sort_values(by='hpVary')
    return dataR

# data1为dataframe变量，为待搜索的成分组合；composite为变动的的元素名；crange为整数变量，为相应的变动范围，例如0到11；
def SearchLargeSETempRangeNi(data1,Vary):
    # 采用穷举法，搜索只有两个成分变动1的合金体系，比较hp的变化，如果>=300,则记录对应的合金成分编号，记录hp的差值
    # 只允许Ni替代Ti
    dataR = pd.DataFrame(columns=['alloy1','alloy2','hpVary'])
    for i in tqdm(range(data1.shape[0])):
        loclist = []
        dataVary = data1.loc[i]-data1
        # 针对dataVary，需要统计的量为行最大值，行零值个数，hp/sum
        dataVary1 = dataVary.drop(['hp'],axis=1)
        listMax = list(dataVary1.max(axis=1))
        dataVary2 = dataVary.replace(0,np.nan)
        listNum0 = dataVary2.isnull().sum(axis=1)
        listHpVary = abs(dataVary.loc[:,'hp']/dataVary1.max(axis=1))
        dataVary.insert(dataVary.shape[1],'listNum0',listNum0)
        dataVary.insert(dataVary.shape[1], 'listMax', listMax)
        dataVary.insert(dataVary.shape[1], 'listHpVary', listHpVary)
        dataVaryWant = dataVary[(dataVary['listNum0']==9)&(dataVary['listHpVary']>=Vary)&(dataVary['listMax']==1)&
                                (np.abs(dataVary['Ti']) ==1)&(np.abs(dataVary['Ni']) ==1)].copy()
        if not dataVaryWant.empty:
            listalloy1 = [m*0+i for m in range(dataVaryWant.shape[0])]
            dataRR = pd.DataFrame()
            dataRR.insert(0, 'alloy1', listalloy1)
            dataRR.insert(1, 'alloy2', dataVaryWant.index.values)
            dataRR.insert(2, 'hpVary', list(dataVaryWant.loc[:,'listHpVary']))
            dataR = pd.concat([dataR,dataRR])
    dataR = dataR.sort_values(by='hpVary')
    return dataR

if __name__ == '__main__':
    order = ['Ti', 'Ni', 'Cu', 'Fe', 'Pd', 'Co', 'Mn', 'Cr', 'Nb', 'Zr', 'Hf']
    path_feature = 'NormalAlloyFeature.csv'
    path_hp = 'SMA.data.training2.csv'


    # 待预测的虚拟空间
    path_Cu = 'data_TiNiHfZrCu.csv'

    data_TiNiHfZrCu = pd.read_csv(filepath_or_buffer=path_Cu,usecols=[i for i in range(1, 12)])
    data_feature = pd.read_csv(filepath_or_buffer=path_feature)
    data_feature = data_feature[order] # 调整data_feature列排布顺序
    Slope = pd.read_pickle('Slope')



    # 生成训练集特征
    data_hp = pd.read_csv(filepath_or_buffer=path_hp, usecols=[i for i in range(1, 13)])
    data_hp2 = pd.read_csv(filepath_or_buffer=path_hp, usecols=[i for i in range(1, 12)])


    data_hp_feature = getPredictDataFeature(data_feature, data_hp2, Slope, m1=100, m2=14)
    data_hp_feature.insert(data_hp_feature.shape[1],'hp',data_hp.loc[:,'hp'].values)
    data_hp_train = pd.concat([data_hp2,data_hp_feature],axis=1)
    data_TiNiHfZrCu_feature = getPredictDataFeature(data_feature, data_TiNiHfZrCu, Slope, m1=100, m2=14)
    model_hp_feature = [('slope_imp', 'ar_c', 'ea'), 'hp']


    # 使用全部的数据库数据（包含部分文献值）预测
    model_hp = GetModel(data_hp_feature, model_hp_feature)
    data_predict_Ap_Cu = GetPredictData(model=model_hp, feature=model_hp_feature, data_train=data_hp_feature,
                                     data_predict=data_TiNiHfZrCu, data_predict_feature=data_TiNiHfZrCu_feature)
    data_predict_hp_Cu = data_predict_Ap_Cu


    # 从预测集中筛选变化剧烈的成分
    # dataRNi_Cu = SearchLargeSETempRange(data_predict_hp_Cu,Vary=300)
    dataRNi_Cu = SearchLargeSETempRangeNi(data_predict_hp_Cu,Vary=50)
    dataRNi_Cu.drop_duplicates(subset='hpVary',keep='first',inplace=True)
    dataRNi_Cu = dataRNi_Cu.reset_index(drop=True)
    dataRRNi_Cu = SearchLargeSETempRange1(dataRNi_Cu, data_predict_hp_Cu)

    dataRRNi_CuFe = dataRRNi_Cu
    dataRRNi_CuFe = dataRRNi_CuFe.drop_duplicates()
    dataRRNi_CuFe.sort_values(by='hpVary', inplace=True, ascending=True)
    dataRRNi_CuFe.sort_values(by='hp1', inplace=True, ascending=True)
    dataRRNi_CuFe = dataRRNi_CuFe.reset_index(drop=True)

    # alloy1和alloy2的字符串长度必须相等
    num = []
    for i in range(dataRRNi_CuFe.shape[0]):
        if len(dataRRNi_CuFe.iloc[i,0]) == len(dataRRNi_CuFe.iloc[i,2]):
            num.append(i)
    dataRRNi_CuFe2 = dataRRNi_CuFe.iloc[num,:].copy()
    # 保证前后的元素种类相等
    element = ['Ti','Ni','Hf','Zr','Cu','Fe']
    num2 = []
    for i in tqdm(range(dataRRNi_CuFe2.shape[0])):
        element1 = []
        element2 = []
        for j in range(len(element)):
            if element[j] in dataRRNi_CuFe2.iloc[i,0]:
                element1.append(element[j])
        for j in range(len(element)):
            if element[j] in dataRRNi_CuFe2.iloc[i,2]:
                element2.append(element[j])
        if element1 == element2:
            num2.append(i)
    dataRRNi_CuFe3 = dataRRNi_CuFe2.iloc[num2,:].copy()
    dataRRNi_CuFe3.sort_values(by='hpVary', inplace=True, ascending=False)
    dataRRNi_CuFe3 = dataRRNi_CuFe3.reset_index(drop=True)


    dataRNi_Cu.reset_index(drop=True,inplace=True)
    # 获取Cu、Hf、Zr的元素含量
    Cu,Hf,Zr,hpVary,Ti,Ni,hp1,hp2 = [],[],[],[],[],[],[],[]
    for i in tqdm(range(dataRNi_Cu.shape[0])):
        if data_predict_hp_Cu.iloc[dataRNi_Cu.iloc[i, 0], 11]>data_predict_hp_Cu.iloc[dataRNi_Cu.iloc[i, 1], 11]:
            inum = 1
        else:
            inum = 0
        Cu.append(data_predict_hp_Cu.iloc[dataRNi_Cu.iloc[i,inum],2])
        Hf.append(data_predict_hp_Cu.iloc[dataRNi_Cu.iloc[i, inum], 10])
        Zr.append(data_predict_hp_Cu.iloc[dataRNi_Cu.iloc[i, inum], 9])
        Ti.append(data_predict_hp_Cu.iloc[dataRNi_Cu.iloc[i, inum], 0])
        Ni.append(data_predict_hp_Cu.iloc[dataRNi_Cu.iloc[i, inum], 1])
        hp1.append(data_predict_hp_Cu.iloc[dataRNi_Cu.iloc[i, 0], 11])
        hp2.append(data_predict_hp_Cu.iloc[dataRNi_Cu.iloc[i, 1], 11])
        hpVary.append(dataRNi_Cu.iloc[i, 2])
    plot_data = pd.DataFrame({'Ti':Ti,'Ni':Ni,'Cu':Cu,'Hf':Hf,'Zr':Zr,'hp1':hp1,'hp2':hp2,'hpVary':hpVary})
    plot_data1 = plot_data[(plot_data['Ni']+plot_data['Cu'])==51].copy()
    plot_data1.reset_index(drop=True, inplace=True)