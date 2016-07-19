from sklearn.feature_extraction import DictVectorizer #为了让sklearn能够正确读取数据,做数据转化
import csv
from sklearn import preprocessing
from sklearn import tree
from sklearn.externals.six import StringIO


#从csv文件数据
with open('/Users/mengxiangcheng/documents/project/python/test/allelectronics.csv', newline='',encoding='utf-8') as csvfile:
 reader = csv.reader(csvfile,delimiter=';', quoting=csv.QUOTE_NONE)  #类似于split(";")
 headers = next(reader)  #读取第一行,要有缩进(空格),否则不在with里会认为关闭了,报I/O operation on closed file 错误

 print(headers)

 featureList = []
 labelList = []

 for row in reader:
     # print(row)
     labelList.append(row[len(row)-1]) #Python len() 方法返回字符串长度。 这一行意思是取每一行最后一个值
     rowDict = {}
     # print(len(row)-1)
     for i in range(1,len(row) - 1):  # 取第二列到倒数第二列中的值 range(1,5) #代表从1到5(不包含5)  range(1,5,2) #代表从1到5，间隔2(不包含5)
         rowDict[headers[i]] = row[i] # 拼接成这种格式{'student': 'no', 'age': 'youth', 'income ': 'high', 'credit_rating': 'fair'}
     featureList.append(rowDict)

 print(featureList)

 vec = DictVectorizer()
 dummyX = vec.fit_transform(featureList) .toarray()

 print(" dummyX" +str(dummyX)) # 比如前三个是age值分别是middle_aged,senior,youth 第一行数据age是youth,所以前三列结果是0,0,1 以此类推
 print(vec.get_feature_names()) #输出的列和str(dummyX) 对应的转换0,1 相呼应,假如这里值是middle_aged 那对应的行列是middle_aged 则是1否则是0

 lb = preprocessing.LabelBinarizer()
 dummyY = lb.fit_transform(labelList)
 print("dummY" + str(dummyY))

 #使用决策树
 clf = tree.DecisionTreeClassifier(criterion='entropy') #声明使用决策树ID3算法
 clf = clf.fit(dummyX,dummyY);
 print("clf" + str(clf))

#生成dot文件
with open("allEletronicInformationGainOri.dot", 'w') as f:
    f = tree.export_graphviz(clf,feature_names=vec.get_feature_names(),out_file=f)

#生成dot文件后再用graphviz工具把这个画出来

#改变第一条数据测试,将第一行年轻人，变成中年人，再做结果输出
oneRowX = dummyX[0, :]
print("oneRowX" + str(oneRowX))

newRowX = oneRowX
newRowX[0] = 1
newRowX[1] = 0
print("newRowX:" + str(newRowX))

#做出结果
predictedY = clf.predict(newRowX)
print("新数据结果" + str(predictedY))


# def demo():
#     s = urllib.urlopen("http://www.zhid58.com")
#     print (s.readlines())
#
#
# if __name__ == '__main__':
#     demo();