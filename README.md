# Supply-Chain-Demand-Inventory-Analytics
## 数据描述
本数据集为沃尔玛销售预测项目提供了一套完整、多层级的历史数据，旨在支持从宏观战略分析到微观库存决策的全方位供应链洞察。数据核心为覆盖美国三州（CA、TX、WI）门店的长周期日度销售记录（d_1 至 d_1913），并通过关联表整合了时间、价格与产品信息。  

数据采用典型的星型结构：主事实表（可由sales_train_validation.csv转换而来的长格式data_long.parquet）记录了“产品-门店-日期”粒度的销量；calendar.csv提供日期维度的星期、假日及特殊事件属性；sell_prices.csv则记录了动态的产品定价信息。产品维度按商品、类别、部门进行了层级划分，门店维度也归属于各州。  

该数据集的核心价值在于其丰富的分析维度与层次。在时间上，超过5年的日度数据可支持趋势分解与季节性建模；在空间上，州、门店的划分便于进行区域对比与网络聚合分析；在产品上，从宽泛的品类（如FOODS）到具体单品（item_id）的层级结构，完美契合“从面到点”的分析漏斗。同时，价格、促销等解释变量的存在，使得需求驱动因素的量化分析成为可能，为精准的需求预测和后续的库存策略模拟（如计算安全库存、再订货点）提供了坚实基础。  
## 分析目标
本次分析旨在运用沃尔玛多层次销售数据，解决供应链管理中“如何将历史销售数据转化为精准、可执行的库存决策”这一核心问题。我们将通过一个“从面到点”的三层漏斗式分析框架，首先识别出具有战略意义的品类与核心销售区域，进而剖析该区域内供应链网络的聚合效应与门店需求特征，最终聚焦于代表性单品与门店，量化其需求规律，并应用库存模型模拟不同策略下的成本与服务权衡，从而为特定商品制定出数据驱动的、最优的安全库存与再订货点建议，实现需求预测到库存策略的闭环落地。
## 数据预处理
本次分析的数据预处理主要使用PYTHON的PANDAS包来完成。首先我们选择FOODS品类进行深度分析，主要基于其核心业务价值与数据典型性。作为高频、刚需的快消品，它是零售基本盘和流量基石，对保障日常运营至关重要。同时，其需求模式相对稳定且规律性强，为我们清晰分离和观察趋势、季节性提供了理想的“典型样本”。更重要的是，食品品类面临的短保质期、高周转特性，使得在服务水平、库存成本与损耗间取得平衡的挑战尤为突出，这使得针对它的库存策略优化最具现实示范意义和业务价值，适合进行供应链研究。数据的导入代码如下：
```python
import pandas as pd
calendar = pd.read_csv(r'D:\供应链\m5-forecasting-accuracy\calendar.csv')
data = pd.read_csv(r'D:\供应链\m5-forecasting-accuracy\sales_train_evaluation.csv')
data=data[data['cat_id']=='FOODS']
prices = pd.read_csv(r'D:\供应链\m5-forecasting-accuracy\sell_prices.csv')
print(calendar.columns)
print(data.columns)
print(prices.columns)
```
为便于进行时间序列分析与聚合计算，我们首先将原始的宽格式销售数据（每个商品-门店组合为一行，1913个日期为列）转换为长格式。使用pd.melt方法，将标识列（如商品、门店、类别等）保留，并将日期列（d_1至d_1913）融合为两列：d（日期编号）和sales（对应日销量）。这一转换使得数据粒度变为“商品-门店-日期”，为后续与日历表和价格表的关联。
```python
#宽表转长表
data_long=pd.melt(data,
    id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],
    value_vars=[f'd_{x}' for x in range(1, 1914)],
    var_name='d',
    value_name='sales'
    )
print(data_long.columns)
print(data_long.head())
```
然后是将三个表连接起来，便于分析和处理，代码如下：
```python
#表连接
data_long=pd.merge(data_long,calendar,on='d')
print(data_long.head())
data_long=pd.merge(data_long,prices,on=['store_id', 'item_id', 'wm_yr_wk'])
print(data_long.head())
```
表中'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2'显示为混合字段，现在将其字段类型统一为字符串，代码如下：
```python
#修改字段类型-混合改字符串
target_columns = ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
for col in target_columns:
    if col in data_long.columns:
        data_long[col] = data_long[col].astype(str)
```
最后，对数据进行去空去重去异常操作，代码如下：
```python
#去空
print(data_long.isnull().sum())
event_cols = ['event_name_1','event_type_1','event_name_2','event_type_2']
for col in event_cols:
    data_long[col].fillna('无', inplace=True)
    data_long[col] = data_long[col].astype('category')
print(data_long.isnull().sum())
#去重
print(data_long.duplicated(subset=['item_id', 'store_id', 'date']).sum())
#data_long.duplicated(subset=['item_id', 'store_id', 'date'],keep='first', inplace=True)
#去异常
print("价格小于等于0的记录:", (data_long['sell_price'] <= 0).sum())
data_long['sales'] = data_long['sales'].clip(lower=0)
print("销量小于等于0的记录:", (data_long['sales'] < 0).sum())
```
## 品类宏观分析
首先对 FOODS 品类进行了宏观层面的趋势与区域分析。  
在趋势分析中，我们将日度销售数据按月份聚合，绘制全国月度销售曲线。该分析旨在清晰地揭示品类的长期增长态势与周期性规律。在区域对比中，我们进一步将销售数据按州（CA, TX, WI）与月份进行分组，通过多系列折线图直观对比不同市场的需求走势，并辅以销售占比饼图量化各州的市场地位。 
<img width="1168" height="660" alt="QQ_1768875141812" src="https://github.com/user-attachments/assets/6cee9abd-fa9e-469e-bc8b-6635828c6c4a" />

