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
根据对 FOODS 品类的宏观分析，我们获得了以下核心洞察：从全国整体趋势来看，该品类销售呈现波动性增长的态势，表明市场总需求在周期性起伏中持续扩张。其需求呈现出显著的季节性规律，每年以7、8、9月为销售峰顶，这很可能与夏季假期、户外活动增加等消费旺季紧密相关。在地域分布上，加利福尼亚州（CA） 无疑是该品类的绝对核心市场，其销售额持续占据全国总量的 42.57%，显著领先于其他各州。这三大结论共同指明，加州不仅是 FOODS 品类的基本盘，也是其季节性增长动能的主要贡献者，因此将其作为后续供应链网络与单品深度分析的重点区域具有充分的战略依据。  
为了从宏观层面评估该品类的供应链管理难度与风险，我们在完成趋势与区域分析后，进行了需求稳定性初探。这一步骤旨在通过计算全国月度销售额的变异系数，量化需求波动的相对幅度，从而对品类整体的需求不确定性做出初步判断。理解这一波动性至关重要，因为它直接决定了供应链需要应对的需求噪音大小，是后续设计库存策略、评估网络聚合效应（即集中库存是否能平滑波动）的基础前提。简言之，此分析将回答一个关键问题：该品类的需求是“相对稳定可控”，还是“波动较大、充满挑战”，为后续深入分析定下基调。
```python
# 1. 确保日期列为日期时间类型
data_long['date'] = pd.to_datetime(data_long['date'])

# 2. 提取“年月”用于月度聚合 (例如 '2011-01')
data_long['year_month'] = data_long['date'].dt.to_period('M')  # 返回Period类型，非常适合分组

# 3. 计算FOODS品类全国月度总销售量
# 筛选品类 & 按月分组求和
monthly_sales_volume = data_long[data_long['cat_id'] == 'FOODS'].groupby('year_month')['sales'].sum()

# 4. 计算关键统计指标：标准差 和 变异系数
std_volume = monthly_sales_volume.std()        # 标准差 (绝对波动)
mean_volume = monthly_sales_volume.mean()      # 均值
cv_volume = std_volume / mean_volume if mean_volume != 0 else 0  # 变异系数 (相对波动)

print(f"月度平均销售量: {mean_volume:.2f}")
print(f"月度销售量标准差: {std_volume:.2f}")
print(f"变异系数 (CV): {cv_volume:.3f}")
```
根据对FOODS品类全国月度销售数据的量化分析，其需求呈现出高度的稳定性。在总计64个月的观察期内，月度平均销售量约为70.5万单位，其波动（标准差约为11.8万）相对有限。关键的衡量指标——变异系数（CV）仅为0.167，远低于0.3的经验阈值。  

这一结果表明，FOODS品类作为一个整体的需求模式可预测性较强。从供应链管理的角度来看，稳定的需求意味着：
更精准的预测：基于历史数据的预测模型将更为可靠。  
更高效的库存配置：在总体层面，为应对不确定性而设置的安全库存水平可以相对较低，从而提升资金效率。  
更稳健的运营基础：为后续聚焦核心区域（CA州）进行网络和单品层面的深入优化，提供了一个波动性较小的分析起点。  
简言之，宏观层面的需求稳定性为实施精细化的库存策略奠定了有利的数据基础。
## 品类区域分析（聚焦CA网络）
### 网络聚合效应验证
网络聚合效应验证旨在为集中式库存管理策略提供定量依据，核心目标是验证“将库存集中于中央仓是否能有效平滑终端需求的波动”这一供应链经典理论。我们通过对比门店层级与中央仓层级的需求波动性来完成验证：首先计算加州每家门店FOODS品类日销量的变异系数（CV），取其平均值代表分散网络的波动水平；然后将所有门店的日销量汇总，计算全加州总日销量的变异系数，代表集中库存所面对的需求波动。通过比较这两个CV值，我们可以量化聚合带来的波动降低幅度，从而从数据上证实或修正理论，为评估网络库存效率、设计补货策略奠定基础。
```python
# 1. 计算每个门店的变异系数
store_daily_sales = data_long.groupby(['store_id', 'date'])['sales'].sum().reset_index()
store_stats = store_daily_sales.groupby('store_id')['sales'].agg(['mean', 'std']).reset_index()
store_stats['cv'] = store_stats['std'] / store_stats['mean']
avg_store_cv = store_stats['cv'].mean()

# 2. 计算中央仓的变异系数
central_daily_sales = store_daily_sales.groupby('date')['sales'].sum().reset_index()
central_mean = central_daily_sales['sales'].mean()
central_std = central_daily_sales['sales'].std()
central_cv = central_std / central_mean

print(f"门店平均变异系数(CV): {avg_store_cv:.3f}")
print(f"中央仓变异系数(CV): {central_cv:.3f}")
```
基于分析，网络聚合效应得到了确凿的量化证实。分析结果显示，加州各门店FOODS品类的平均需求变异系数(CV)为0.241，而将全州需求汇总至中央仓后，其变异系数降低至0.212。这意味着集中库存使得需求波动相对降低了12.1%，清晰地验证了“风险共担”原理：聚合分散且不完全相关的需求，可以有效平滑整体波动，从而为中央仓实施更精准、更高效的库存策略提供了数据基础。

更进一步，将此结果与第一阶段的全国性分析对比，能揭示出供应链网络层级对波动平滑的关键影响。第一阶段计算出的全国月度需求CV为0.167，显著低于加州的中央仓CV(0.212)。这一差异说明，聚合的层级越高（从州级到全国），所能吸纳和抵消的局部随机波动就越多，整体需求将变得更加稳定和可预测。这一对比强有力地证明，构建多层级、集中化的供应链网络，是应对需求不确定性、提升运营稳健性的核心战略。
