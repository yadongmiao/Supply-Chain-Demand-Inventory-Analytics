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
### 趋势与区域分析
首先对 FOODS 品类进行了宏观层面的趋势与区域分析。  
在趋势分析中，我们将日度销售数据按月份聚合，绘制全国月度销售曲线。该分析旨在清晰地揭示品类的长期增长态势与周期性规律。在区域对比中，我们进一步将销售数据按州（CA, TX, WI）与月份进行分组，通过多系列折线图直观对比不同市场的需求走势，并辅以销售占比饼图量化各州的市场地位。 
<img width="1168" height="660" alt="QQ_1768875141812" src="https://github.com/user-attachments/assets/6cee9abd-fa9e-469e-bc8b-6635828c6c4a" />
根据对 FOODS 品类的宏观分析，我们获得了以下核心洞察：从全国整体趋势来看，该品类销售呈现波动性增长的态势，表明市场总需求在周期性起伏中持续扩张。其需求呈现出显著的季节性规律，每年以7、8、9月为销售峰顶，这很可能与夏季假期、户外活动增加等消费旺季紧密相关。在地域分布上，加利福尼亚州（CA） 无疑是该品类的绝对核心市场，其销售额持续占据全国总量的 42.57%，显著领先于其他各州。这三大结论共同指明，加州不仅是 FOODS 品类的基本盘，也是其季节性增长动能的主要贡献者，因此将其作为后续供应链网络与单品深度分析的重点区域具有充分的战略依据。
### 稳定性分析
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
### 门店需求画像分析
门店需求画像分析旨在将宏观的市场洞察转化为具体、可执行的网络管理策略，其核心目标是深入理解FOODS品类在加州供应链网络中的分布特征与运营挑战。我们通过对每个门店进行多维度的量化评估，构建从规模、稳定性到战略重要性的完整画像。具体做法是：首先聚合门店日级销售数据，计算关键指标如总销售额、日均销量及衡量波动性的变异系数；随后运用帕累托法则进行ABC分类，锁定贡献70%销售额的A类核心门店；同时通过四象限分析法，以销量和波动性为维度将门店划分为“高销量低波动”等类型。这一分析不仅为验证网络聚合效应提供了微观基础，更为后续单品级库存优化筛选出具有代表性的目标门店，实现从“知道哪里的市场重要”到“明白具体该如何管理”的决策跨越。具体实现代码如下：
```python
# 2. 计算销售额（销量 × 价格）
data_long['revenue'] = data_long['sales'] * data_long['sell_price']

# 3. 门店日级聚合（计算每个门店每天的销量和销售额）
store_daily = data_long.groupby(['store_id', 'date']).agg({
    'sales': 'sum',  # 总销量
    'revenue': 'sum'  # 总销售额
}).reset_index()


# 4. 计算门店核心指标
def calculate_store_metrics(store_daily_df):
    """计算每个门店的核心指标"""

    # 按门店分组计算
    store_metrics = store_daily_df.groupby('store_id').agg({
        'sales': ['count', 'sum', 'mean', 'std'],
        'revenue': ['sum', 'mean', 'std']
    }).reset_index()

    # 展平多级列名
    store_metrics.columns = [
        'store_id',
        'days_count',  # 观测天数
        'total_quantity',  # 总销量
        'avg_daily_quantity',  # 日均销量
        'std_daily_quantity',  # 日销量标准差
        'total_revenue',  # 总销售额
        'avg_daily_revenue',  # 日均销售额
        'std_daily_revenue'  # 日销售额标准差
    ]

    # 计算变异系数（CV）
    store_metrics['cv_quantity'] = np.where(
        store_metrics['avg_daily_quantity'] > 0,
        store_metrics['std_daily_quantity'] / store_metrics['avg_daily_quantity'],
        0
    )

    store_metrics['cv_revenue'] = np.where(
        store_metrics['avg_daily_revenue'] > 0,
        store_metrics['std_daily_revenue'] / store_metrics['avg_daily_revenue'],
        0
    )

    return store_metrics


store_metrics = calculate_store_metrics(store_daily)


# 5. ABC分类（基于总销售额）
def abc_classification(df, revenue_col='total_revenue', a_threshold=0.7):
    """基于销售额进行ABC分类"""
    df_sorted = df.sort_values(revenue_col, ascending=False).copy()
    df_sorted['cumulative_revenue'] = df_sorted[revenue_col].cumsum()
    total_revenue = df_sorted[revenue_col].sum()
    df_sorted['cumulative_pct'] = df_sorted['cumulative_revenue'] / total_revenue

    # A类：累计占比 ≤ 70%
    # B类：累计占比 70% - 90%
    # C类：累计占比 > 90%
    df_sorted['abc_class'] = 'C'
    df_sorted.loc[df_sorted['cumulative_pct'] <= 0.9, 'abc_class'] = 'B'
    df_sorted.loc[df_sorted['cumulative_pct'] <= 0.7, 'abc_class'] = 'A'

    return df_sorted


store_metrics = abc_classification(store_metrics)


# 6. 四象限分析（基于日均销量和销量变异系数）
def quadrant_analysis(df, x_col='avg_daily_quantity', y_col='cv_quantity'):
    """基于销量和变异系数进行四象限分析"""
    df = df.copy()

    # 计算中位数作为分割点
    x_median = df[x_col].median()
    y_median = df[y_col].median()

    # 四象限分类
    conditions = [
        (df[x_col] > x_median) & (df[y_col] > y_median),  # 第一象限：高销量、高波动
        (df[x_col] > x_median) & (df[y_col] <= y_median),  # 第二象限：高销量、低波动
        (df[x_col] <= x_median) & (df[y_col] > y_median),  # 第三象限：低销量、高波动
        (df[x_col] <= x_median) & (df[y_col] <= y_median)  # 第四象限：低销量、低波动
    ]

    choices = ['高销量高波动', '高销量低波动', '低销量高波动', '低销量低波动']
    df['quadrant'] = np.select(conditions, choices, default='未知')

    # 添加中位数参考值
    df[f'{x_col}_median'] = x_median
    df[f'{y_col}_median'] = y_median

    return df


store_metrics = quadrant_analysis(store_metrics)


# 7. 计算门店间相关性矩阵（用于分析牛鞭效应风险）
def calculate_correlation_matrix(store_daily_df):
    """计算门店日销售额的相关系数矩阵"""
    # 创建门店日销售额的透视表
    revenue_pivot = store_daily_df.pivot_table(
        index='date',
        columns='store_id',
        values='revenue',
        aggfunc='sum'
    ).fillna(0)

    # 计算相关系数矩阵
    correlation_matrix = revenue_pivot.corr()

    # 提取每对门店的相关性
    correlation_list = []
    store_ids = correlation_matrix.columns.tolist()

    for i in range(len(store_ids)):
        for j in range(i + 1, len(store_ids)):
            store1 = store_ids[i]
            store2 = store_ids[j]
            corr_value = correlation_matrix.loc[store1, store2]

            correlation_list.append({
                'store_1': store1,
                'store_2': store2,
                'correlation': corr_value
            })

    correlation_df = pd.DataFrame(correlation_list)

    # 标记高度相关（>0.7）和中度相关（>0.3）的门店对
    correlation_df['correlation_level'] = '低相关(<0.3)'
    correlation_df.loc[correlation_df['correlation'] > 0.3, 'correlation_level'] = '中度相关(0.3-0.7)'
    correlation_df.loc[correlation_df['correlation'] > 0.7, 'correlation_level'] = '高度相关(>0.7)'

    return correlation_matrix, correlation_df


correlation_matrix, correlation_pairs = calculate_correlation_matrix(store_daily)


# 8. 代表性门店筛选
def select_representative_stores(store_metrics_df):
    """基于多个维度筛选代表性门店"""

    # 筛选A类门店
    a_stores = store_metrics_df[store_metrics_df['abc_class'] == 'A']

    # 不同类型代表门店
    representatives = {
        '高销量代表': a_stores.loc[a_stores['total_revenue'].idxmax(), 'store_id'],
        '高波动代表': a_stores.loc[a_stores['cv_quantity'].idxmax(), 'store_id'],
        '低波动代表': a_stores.loc[a_stores['cv_quantity'].idxmin(), 'store_id'],
        '四象限高销量低波动': store_metrics_df[
                                  store_metrics_df['quadrant'] == '高销量低波动'
                                  ].loc[:, 'store_id'].iloc[0] if not store_metrics_df[
            store_metrics_df['quadrant'] == '高销量低波动'
            ].empty else None
    }

    # 创建代表性门店列表
    rep_df = pd.DataFrame({
        'store_id': list(representatives.values()),
        'representative_type': list(representatives.keys()),
        'selection_reason': [
            'A类门店中总销售额最高',
            'A类门店中需求波动最大（变异系数最高）',
            'A类门店中需求波动最小（变异系数最低）',
            '四象限分析中的高销量低波动门店'
        ]
    })

    return rep_df


representative_stores = select_representative_stores(store_metrics)
```
#### 核心结论：四象限门店画像

分析发现，加州四家门店在“销售规模-需求稳定性”维度上差异显著，可被清晰划分至四个管理象限，其特征与业务定位如下表所示：

| 门店ID | 象限定位 | 日均销量(件) vs 中位数 | 需求波动(CV) vs 中位数 | ABC类别 | 业务画像与核心特征 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **CA_3** | **第二象限：高销量、低波动** | 3,929 **≫** 2,316 | 0.205 **<** 0.227 | A类 | **“现金牛门店”**：销量最高且需求最稳定，是网络的利润基石与效率标杆，可预测性极强。 |
| **CA_1** | **第一象限：高销量、高波动** | 2,813 **>** 2,316 | 0.249 **>** 0.227 | A类 | **“问题明星门店”**：销售额贡献第二，但需求波动性突出，是库存管理与预测优化的关键挑战点。 |
| **CA_2** | **第三象限：低销量、高波动** | 1,819 **<** 2,316 | 0.317 **≫** 0.227 | B类 | **“风险门店”**：销量有限且需求极不稳定，预测误差大，属于高管理成本、低收益的类型。 |
| **CA_4** | **第四象限：低销量、低波动** | 1,474 **<** 2,316 | 0.194 **<** 0.227 | C类 | **“长尾门店”**：销量小但需求平稳，运营简单，管理成本低。 |

**分析**：该画像表明，加州的供应链网络并非同质化，而是由需求模式迥异的门店组成。若采用统一的库存策略，将无法匹配各店实际风险与需求特点，必然导致在CA_1、CA_2等店面临高缺货或高库存风险，而在CA_3、CA_4等店则可能存在资源浪费。因此，**差异化、精细化的库存策略是管理该网络的核心**。

### 网络聚合效应验证与关键发现

通过计算门店需求间的相关性，我们发现了影响供应链网络效率的一个关键特性：

**关键数据**：门店间日销售额的相关系数矩阵显示，网络中存在强烈的**需求同步性**。特别是CA_1、CA_3、CA_4三店之间，相关系数均高于0.84，全部门店的平均相关系数达**0.75**，属于高度正相关。

**分析与启示**：这一发现直接解释了为何经典的“风险共担”（集中库存以平滑波动）效应在本网络中表现不显，即聚合后中央仓的CV（0.212）比门店平均CV（0.241）有所降低（得益于规模基数增大），但降低幅度（12.1%）远低于理想中独立需求下的理论值。甚至可能出现“聚合后波动不减反增”（标准差）的反直觉现象。其根本原因在于，当所有门店的需求受相同因素（如全州促销、节假日、天气）驱动而**同涨同跌**时，在中央仓层面汇总需求无法有效抵消波动。相反，波动会被叠加并放大。
> **管理启示**：对于需求高度正相关的网络，简单地集中库存至中央仓所带来的效益有限。供应链弹性的提升应侧重于**建立更灵活的响应机制**（如敏捷的运输能力）和**制定基于门店画像的差异化库存参数**，而非完全依赖集中库存来降低安全库存水平。
#### 供应链管理策略建议

基于门店画像与网络特性，提出以下差异化策略：

1.  **门店级库存策略**：
    *   **CA_3 (现金牛)**：实施**高效自动化补货**。可采用较低的**安全库存系数**和固定的**再订货点（ROP）**，目标是最大化周转率与成本效率，并将其策略作为网络标杆。
    *   **CA_1 (问题明星)**：实行**重点监控与动态管理**。必须配置更高的**安全库存**，并采用更短周期的预测回顾（如每周），可考虑引入更高级的预测模型（如结合促销事件）。此门店是提升整体网络服务水平的关键。
    *   **CA_2 (风险门店)**：采取**保守的成本控制策略**。优先考虑按订单补货或设置极低的安全库存，以规避库存积压风险，接受较低的服务水平。
    *   **CA_4 (长尾门店)**：推行**简化管理**。采用周期性的集中复查与补货（如每两周一次），减少日常管理投入。

2.  **网络级运营优化**：
    *   在补货计划中，应根据门店的**CV值（波动性）** 设定差异化的库存覆盖天数，而非“一刀切”。
    *   在物流配送上，可对CA_1和CA_3采用更频繁的配送班次，以快速响应需求变化并降低平均库存水平。
#### 下一阶段分析对象选定

为进行第三阶段的单品级库存优化模拟，需要选择最具代表性的门店。推荐选择以下两家，以覆盖最具挑战性和最典型的业务场景：

1.  **CA_1**：作为 **“高销量、高波动”** 的典型代表。对其的分析能直接解决供应链中最棘手的问题，得出的安全库存与再订货点建议将极具现实指导意义。
2.  **CA_3**：作为 **“高销量、低波动”** 的效率标杆。对其的分析能确立成本最优的库存策略基准，为网络其他门店提供优化方向。 
此选择确保了后续单品分析结论既能应对最大挑战，又能确立效率标准，从而形成从宏观到微观的完整决策闭环。


## 单品微观诊断（聚焦CA区域，特定门店与单品）
### 单品筛选与数据准备
本步骤旨在将第二阶段的网络分析结论具体化，从选定的两家代表性门店（CA_1与CA_3）中，锁定少数几个核心单品作为微观诊断的最终对象。我们依据帕累托法则（80/20法则），分别在每家门店内部，按照商品的历史总销售额进行排序，筛选出排名前二的单品。这种方法确保了我们所分析的对象是驱动该门店业绩的核心商品，其库存策略的优化能产生最大的业务影响。最终，我们从两家门店共筛选出4个代表性单品，并生成了包含其ID、销售额、日均销量及排名信息的清单。此步骤的产出为后续的深度分析提供了明确的目标，实现了分析焦点从“门店”到“具体商品”的精准落地。
```python
# 步骤1：单品筛选与数据准备
# 从CA_1和CA_3两家门店中筛选销售额TOP2单品
# ============================================================================
print("\n" + "=" * 60)
print("步骤1：单品筛选与数据准备")
print("=" * 60)

# 定义目标门店
target_stores = ['CA_1', 'CA_3']

# 筛选目标门店数据
target_stores_data = data_long[data_long['store_id'].isin(target_stores)].copy()
print(f"目标门店数据行数: {len(target_stores_data)}")

# 计算每个门店每个单品的销售表现
item_performance = target_stores_data.groupby(['store_id', 'item_id']).agg({
    'sales': ['sum', 'mean', 'std', 'count'],
    'revenue': 'sum'
}).reset_index()

# 展平多级列名
item_performance.columns = [
    'store_id', 'item_id',
    'total_quantity', 'avg_daily_quantity', 'std_daily_quantity', 'sales_days',
    'total_revenue'
]

# 计算变异系数
item_performance['cv_quantity'] = item_performance['std_daily_quantity'] / item_performance['avg_daily_quantity']
item_performance['cv_quantity'] = item_performance['cv_quantity'].replace([np.inf, -np.inf], 0).fillna(0)

# 对每个门店按总销售额排序，选取TOP2单品
top_items_by_store = []

for store in target_stores:
    store_items = item_performance[item_performance['store_id'] == store].copy()
    store_items = store_items.sort_values('total_revenue', ascending=False)

    # 取前2名
    store_top2 = store_items.head(2).copy()

    # 添加排名信息
    store_top2['store_rank'] = range(1, len(store_top2) + 1)
    store_top2['selection_reason'] = f'在{store}门店销售额排名第' + store_top2['store_rank'].astype(str)

    top_items_by_store.append(store_top2)

# 合并结果
target_items_df = pd.concat(top_items_by_store, ignore_index=True)

# 保存结果
target_items_file = os.path.join(output_dir, 'target_items.csv')
target_items_df.to_csv(target_items_file, index=False, encoding='utf-8-sig')
print(f"单品筛选结果已保存至: {target_items_file}")
print(f"共筛选出 {len(target_items_df)} 个代表性单品")

# 显示筛选结果
print("\n筛选的单品列表:")
for idx, row in target_items_df.iterrows():
    print(f"{row['store_id']} - {row['item_id']}: "
          f"总销售额${row['total_revenue']:.2f}, "
          f"日均销量{row['avg_daily_quantity']:.2f}, "
          f"排名{row['store_rank']}")
```
在单品筛选阶段，我们发现一个极具洞察力的结果：原计划从CA_1与CA_3门店分别筛选销售额排名前二的单品，但实际选出的四个席位被**相同的两个单品所占据**，最终仅得到两个独特的分析对象。

这一现象是前一阶段“门店需求高度相关”结论的强力微观佐证。它表明，这两家门店不仅整体需求波动同步，其**核心销售引擎（即最重要的畅销单品）也完全一致**。这超越了偶然性，指向更深层的业务现实：第一，两家门店所服务的**顾客群体与消费偏好高度相似**；第二，公司的**商品策略、采购与配送计划可能在网络层面高度统一**，导致各门店的主力商品结构趋同。

从供应链管理视角看，此结果具有双重含义：
1.  **积极面（简化与聚焦）**：核心商品的高度集中，极大地简化了网络层面的库存规划。中央仓可以将预测、备货与补货资源聚焦于这少数几个“超级单品”上，通过规模效应降低采购与持有成本，实现管理效率的显著提升。
2.  **风险面（脆弱性增加）**：这也意味着整个网络的销售业绩过度依赖于少数几个单品。一旦这些单品出现断货、质量问题或市场需求骤变，将同时冲击两家主力门店的业绩，使供应链网络面临更高的集中性风险。

因此，这一筛选结果不仅锁定了微观分析的目标，更警示我们：在享受集中化管理带来的效率红利时，必须为这些核心单品设置更稳健的安全库存，并考虑适度培育门店间差异化的长尾商品，以增强供应链的整体韧性。
1. 销售额排名背后的业务逻辑
数据：在两个门店中，FOODS_3_120 的总销售额均高于 FOODS_3_090（如CA_3店：44.2万 vs 33.9万），但 FOODS_3_090 的总销量和日均销量却远高于前者（CA_3店：25万件 vs 8.9万件）。

分析：这表明 FOODS_3_120 是一个高单价、低流量的“利润型”商品，而 FOODS_3_090 是低单价、高流量的“流量型”商品。两者以不同的方式成为门店支柱，决定了库存策略需差异化：前者更关注单品利润与损耗，后者更关注周转率与缺货成本。
综合策略画像与下一步模拟方向
对于 FOODS_3_090 (高流量型)：

策略重心：平衡缺货损失与库存持有成本。因其流量大，缺货的机会成本高，但同样因其销量大，库存绝对值高，持有成本也高。

模拟关键：在第三阶段的成本权衡分析中，应为其设定一个相对较高的单位缺货损失(C_s)。

对于 FOODS_3_120 (高利润型)：

策略重心：保障利润与控制损耗。高单价意味着更高的资金占用和潜在的贬值损耗风险。

模拟关键：应为其设定一个相对较高的单位持有成本(H)，以反映其更高的资金成本和风险。
### 单品需求深度分析
在确定目标单品后，本步骤对其进行严格的需求规律量化，为构建库存模型夯实数据基础。我们计算了每个单品历史日需求的核心统计参数，包括均值(μ)、标准差(σ)和变异系数(CV)，这些是计算安全库存与再订货点的关键输入。同时，我们生成了其需求分布的直方图数据，并拟合了对应的正态分布曲线，以直观评估其需求波动形态是否符合经典库存模型的假设。此外，分析还涵盖了分位数、零销量天数比例等指标，全面刻画需求的集中趋势、离散程度与分布特征。这一步将单品抽象的历史销售记录转化为一系列可运算的、具有明确管理意义的参数，为下一步的库存策略模拟与优化提供了可靠的“事实依据”。
```python
# 步骤2：单品需求深度分析
# 对每个选定的单品进行详细的需求分析
# ============================================================================
print("\n" + "=" * 60)
print("步骤2：单品需求深度分析")
print("=" * 60)

# 获取选定单品的详细信息
selected_items_data = []

for idx, row in target_items_df.iterrows():
    store_id = row['store_id']
    item_id = row['item_id']

    # 筛选该单品在该门店的所有数据
    item_data = data_long[
        (data_long['store_id'] == store_id) &
        (data_long['item_id'] == item_id)
        ].copy()

    # 计算基本统计量
    daily_sales = item_data['sales']
    mu = daily_sales.mean()
    sigma = daily_sales.std()
    cv = sigma / mu if mu > 0 else 0

    # 计算分位数
    q25 = daily_sales.quantile(0.25)
    median = daily_sales.median()
    q75 = daily_sales.quantile(0.75)

    # 计算零销售天数比例
    zero_sales_days = (daily_sales == 0).sum()
    zero_sales_ratio = zero_sales_days / len(daily_sales)


    # 计算最大连续销售/非销售天数
    def max_consecutive_days(series, value=0):
        max_count = 0
        current_count = 0
        for val in series:
            if val == value:
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0
        return max_count


    max_consecutive_zero = max_consecutive_days(daily_sales, 0)
    max_consecutive_sales = max_consecutive_days(daily_sales > 0, True)

    # 收集单品统计信息
    item_stats = {
        'store_id': store_id,
        'item_id': item_id,
        'mu': mu,
        'sigma': sigma,
        'cv': cv,
        'q25': q25,
        'median': median,
        'q75': q75,
        'min': daily_sales.min(),
        'max': daily_sales.max(),
        'total_days': len(daily_sales),
        'zero_sales_days': zero_sales_days,
        'zero_sales_ratio': zero_sales_ratio,
        'max_consecutive_zero': max_consecutive_zero,
        'max_consecutive_sales': max_consecutive_sales,
        'store_rank': row['store_rank']
    }

    selected_items_data.append(item_stats)

# 创建单品需求统计表
item_demand_stats = pd.DataFrame(selected_items_data)

# 保存单品需求统计
stats_file = os.path.join(output_dir, 'item_demand_stats.csv')
item_demand_stats.to_csv(stats_file, index=False, encoding='utf-8-sig')
print(f"单品需求统计数据已保存至: {stats_file}")

# 显示统计摘要
print("\n单品需求统计摘要:")
print(item_demand_stats[['store_id', 'item_id', 'mu', 'sigma', 'cv', 'zero_sales_ratio']].round(4))

# ============================================================================

```
基于 `item_demand_stats.csv` 的分析结果，我们对选定的两个核心单品（`FOODS_3_120` 和 `FOODS_3_090`）在两家门店的需求模式进行了深度量化。以下是针对各项关键指标的逐一解读与分析：

### 一、需求水平分析：衡量“通常卖多少”

1.  **日均需求量 (`mu`)**：
    *   **数据**：所有单品的 `mu` 均大于0，表明存在持续需求。`FOODS_3_090` 的需求量显著高于 `FOODS_3_120`。尤其在CA_3门店，`FOODS_3_090` 的日均需求高达130.9件，是CA_1门店同单品（66.5件）的近两倍，也远高于其他组合。
    *   **分析**：这验证了CA_3作为“高销量”门店的定位。`FOODS_3_090` 是绝对的销量明星，尤其在CA_3店，是需重点保障供应的核心商品。

2.  **需求中位数 (`median`)**：
    *   **数据**：各单品的中位数与均值 (`mu`) 较为接近，但普遍略低于均值（例如CA_1的 `FOODS_3_120`，中位数40件，均值39.96件）。
    *   **分析**：中位数与均值接近，说明需求分布**没有受到极端极高的异常值严重扭曲**。中位数略低于均值，暗示分布有轻微右偏（即存在一些销量特别高的日子，将平均值拉高）。

### 二、需求波动性分析：衡量“预测有多难”

3.  **需求标准差 (`sigma`)**：
    *   **数据**：所有单品的标准差 (`sigma`) 数值都非常高，几乎与甚至超过其日均需求 (`mu`)。例如，CA_1店的 `FOODS_3_090`，`sigma` (57.86) 接近 `mu` (66.49)。
    *   **分析**：**绝对波动幅度极大**。这意味着每日实际销量可能大幅偏离平均水平，给日常备货带来巨大挑战。

4.  **变异系数 (`cv`)**：
    *   **数据**：所有单品的 `cv` 值均在 **0.77 至 0.87** 之间，远超0.5的“高波动”阈值。
    *   **分析**：这是**最关键的风险指标**。它表明需求的**相对波动性极高，可预测性非常差**。经典库存模型中基于稳定需求的假设在此将严重失效。必须为此类单品设置极高的安全库存或采用更敏捷的补货策略。

### 三、需求分布分析：揭示“销售的具体形态”

5.  **分位数 (`q25`, `median`, `q75`)**：
    *   **数据**：以CA_3店的 `FOODS_3_090` 为例，25%的日子销量≤28件，50%的日子销量≤126件（中位数），75%的日子销量≤189件。
    *   **分析**：**四分位距（IQR = q75 - q25）非常大**（如CA_3店的 `FOODS_3_090` 为161件），直观证实了需求的巨大离散程度。大部分日销量落在非常宽的区间内。

6.  **极值 (`min`, `max`)**：
    *   **数据**：所有单品的最小值(`min`)均为0，最大值(`max`)极高（如CA_3店 `FOODS_3_090` 高达763件）。
    *   **分析**：存在**零销量日**和**爆发式销售日**。峰值销量是日均销量的数倍至十余倍（如CA_1店 `FOODS_3_090` 峰值是均值的9倍），这种“脉冲式”需求是导致高波动性的直接原因。

### 四、零销售模式分析：识别“滞销风险”

7.  **零销售天数与比例 (`zero_sales_days`, `zero_sales_ratio`)**：
    *   **数据**：零销售比例在18.7%至24.5%之间。即，每年约有 **68 至 89 天** 完全卖不动。
    *   **分析**：这不是偶尔缺货，而是**一种常态化的间歇性需求模式**。高零销比例与高峰值并存，表明该商品可能属于**促销驱动型**或**季节性/场合性消费商品**，非日常稳定消耗品。

8.  **最大连续零销售天数 (`max_consecutive_zero`)**：
    *   **数据**：极其惊人，所有单品都曾连续超过 **116 天** 甚至 **229 天** 无销售。
    *   **分析**：这是**最危险的信号**。它意味着，若在销售淡季或周期内错误地堆积库存，将面临**长达数月**的库存冻结和资金占用风险，远超商品可能保质期。这强烈反对采用基于平均需求的常规补货模型。

### 综合对比与业务结论

*   **门店对比 (CA_3 vs CA_1)**：CA_3店在销量 (`mu`) 上全面碾压CA_1店，且波动性 (`cv`) 相对略低，零销售比例也稍低。这印证了CA_3是“高效现金牛”，而CA_1是“高波动问题明星”的画像。
*   **单品对比 (`FOODS_3_090` vs `FOODS_3_120)**：`FOODS_3_090` 是更大规模的“明星”，但也是更大风险的来源。其销量和波动绝对值都更高，管理难度更大。
*   **核心发现**：这两个所谓的“核心畅销品”，其需求本质是**高频、高波动、间歇性、脉冲式**的。它们贡献大量销售额，但**绝非稳定流动的“常青树”商品**。

**对库存策略的颠覆性启示**：
传统的、基于常态分布和连续补货的模型在此几乎不适用。必须考虑：
1.  **事件驱动型预测**：需求可能与促销、节假日强相关，需结合日历数据建模。
2.  **高低库存的双重管理**：需准备在促销期承受极高库存，同时在漫长淡季将库存降至极低水平。
3.  **强调可视性与响应速度**：供应链必须具备极高的可视性以预测“脉冲”，并具备快速响应能力以在需求来临前备货，结束后清仓。

这份数据画像没有给出简单的答案，而是清晰地揭示了现实业务的复杂性，并指明了下一步模拟优化必须遵循的特殊方向。
```python
# 生成直方图数据（用于Power BI绘制需求分布图）
# ============================================================================
print("\n" + "=" * 60)
print("生成需求分布直方图数据")
print("=" * 60)

histogram_data_list = []

for idx, row in target_items_df.iterrows():
    store_id = row['store_id']
    item_id = row['item_id']

    # 筛选数据
    item_data = data_long[
        (data_long['store_id'] == store_id) &
        (data_long['item_id'] == item_id)
        ]

    daily_sales = item_data['sales']

    # 自动确定合适的直方图分箱
    # 使用Sturges规则确定分箱数量
    n = len(daily_sales)
    if n > 0:
        num_bins = int(np.ceil(np.log2(n)) + 1)
        num_bins = min(num_bins, 30)  # 限制最大分箱数

        # 计算直方图
        counts, bin_edges = np.histogram(daily_sales, bins=num_bins)

        # 为每个分箱创建数据
        for i in range(len(counts)):
            bin_start = bin_edges[i]
            bin_end = bin_edges[i + 1]
            bin_center = (bin_start + bin_end) / 2

            histogram_data_list.append({
                'store_id': store_id,
                'item_id': item_id,
                'bin_start': bin_start,
                'bin_end': bin_end,
                'bin_center': bin_center,
                'frequency': counts[i],
                'relative_frequency': counts[i] / n
            })

        print(f"为 {store_id}-{item_id} 生成了 {num_bins} 个分箱")

# 创建直方图数据表
histogram_df = pd.DataFrame(histogram_data_list)

# 保存直方图数据
histogram_file = os.path.join(output_dir, 'item_demand_histogram.csv')
histogram_df.to_csv(histogram_file, index=False, encoding='utf-8-sig')
print(f"直方图数据已保存至: {histogram_file}")
print(f"直方图数据行数: {len(histogram_df)}")

# ============================================================================
# 生成正态分布参考线数据（用于在Power BI中叠加正态分布曲线）
# ============================================================================
print("\n" + "=" * 60)
print("生成正态分布参考线数据")
print("=" * 60)

norm_dist_data_list = []

for idx, row in item_demand_stats.iterrows():
    store_id = row['store_id']
    item_id = row['item_id']
    mu = row['mu']
    sigma = row['sigma']

    # 如果标准差为0，跳过
    if sigma == 0 or pd.isna(sigma):
        continue

    # 生成正态分布x值范围 (μ ± 4σ)
    x_min = max(0, mu - 4 * sigma)
    x_max = mu + 4 * sigma

    # 生成100个点
    x_values = np.linspace(x_min, x_max, 100)

    # 计算正态分布概率密度
    from scipy.stats import norm

    y_values = norm.pdf(x_values, mu, sigma)

    # 创建数据
    for x, y in zip(x_values, y_values):
        norm_dist_data_list.append({
            'store_id': store_id,
            'item_id': item_id,
            'x_value': x,
            'norm_density': y,
            'mu': mu,
            'sigma': sigma
        })

    print(f"为 {store_id}-{item_id} (μ={mu:.2f}, σ={sigma:.2f}) 生成了正态分布参考线")

# 创建正态分布数据表
norm_dist_df = pd.DataFrame(norm_dist_data_list)

# 保存正态分布数据
norm_dist_file = os.path.join(output_dir, 'item_norm_distribution.csv')
norm_dist_df.to_csv(norm_dist_file, index=False, encoding='utf-8-sig')
print(f"正态分布参考线数据已保存至: {norm_dist_file}")
```
需求直方图与正态分布曲线拟合不佳——是本次单品诊断中最具业务洞察力的关键发现。这一现象并非分析缺陷，而是对数据内在特性的最真实反映：它直观印证了极高的变异系数（CV>0.77）和近20%的零销日所揭示的间歇性、脉冲式需求模式。图形很可能呈现显著的“零值堆积”和右偏“长尾”，这与经典库存模型所依赖的稳定、连续的正态分布假设根本不符。

因此，这一发现彻底颠覆了后续库存策略的设计前提。它明确警示，直接套用基于正态假设的报童模型来计算安全库存将严重脱离实际。对于 FOODS_3_090 与 FOODS_3_120 这类商品，管理核心必须从“寻求最优公式参数”转向“设计弹性应对机制”。这意味着库存策略应分离为日常低水位补货与事件驱动的高水位备货双轨模式，且必须与营销日历深度耦合。最终的策略建议将不是一组固定的再订货点，而是一套根据预测销售事件进行动态切换的库存管理规则。

基于深度分析，我们识别出导致单品需求呈现高波动性（CV>0.77）与分布拟合不佳的核心症结：少数由促销或节假日驱动的“极端日期”所产生的爆发性销量。这些日期不仅极大地拉高了需求的标准差，其与大量零销日共存的模式，更直接导致直方图出现“零值堆积”与“右偏长尾”的扭曲形态。这并非经典库存模型的根本性失效，而是揭示了其静态参数体系在应对结构性需求突变时存在局限。具体而言，模型在管理长期、平稳的“基线需求”时依然有效，但无法自适应地处理偶发、高影响的“脉冲需求”。

因此，后续的库存策略不应抛弃经典框架，而需对其进行关键增强，构建一个 “基线+事件”的双轨自适应系统。该系统以经典模型计算出的安全库存与再订货点作为管理日常稳态运营的基准；同时，必须深度集成日期分析，通过识别历史与计划中的营销事件，主动在“极端日期”来临前触发库存预案，动态叠加或切换至更高的“事件安全库存”水平。这一策略的本质，是将供应链响应从被动应对波动，升级为主动管理已知的业务节奏。
### 参数设置
在参数与假设设定阶段，我们严格遵循“基于业务逻辑、适配数据特性、支持管理决策”的原则。所有关键参数均非随意指定，其背后是行业经验与数据特征的结合：

补货提前期 (L=2天)：采用零售杂货行业对高频补货的常规假设。这是一个基准值，管理者可根据实际物流效率进行替换，模型将自动重新计算所有库存参数。

服务水平范围与Z值 (80% 至 99%)：选择这一范围旨在覆盖从“成本优先”到“服务优先”的完整管理策略光谱。特别地，我们纳入了95%以上区间的分析，因为对于CV值极高的单品，服务水平每提升1个百分点，所要求的安全库存增量将急剧扩大，这能为管理层揭示边际效益的拐点。

成本参数的核心逻辑：

持有成本率 (年化20%)：此为零售库存的典型综合成本，包含资金占用（约8-10%）、仓储运营、保险及商品损耗风险。

缺货成本系数 (3倍毛利)：这是为应对高波动、脉冲式需求特性的关键假设。对于此类商品，缺货不仅损失单笔交易利润，更可能因促销机会错失或顾客体验下降而带来数倍于毛利的长期隐性损失。该系数旨在量化此放大效应。

动态成本计算：模型并非使用统一的成本绝对值，而是基于每个单品的实际历史平均售价来动态计算其单位持有成本与缺货成本。这确保了成本分析能反映商品间的价值差异。

模拟的再订货点范围：成本模拟不仅测试了由报童模型生成的几个理论点，更围绕日均需求 (μ) 设置了从50%到300%的宽泛测试区间。这一设计的目的是确保在U形成本曲线上，能完整捕捉到最低成本点，避免因测试点不足而错过最优策略。

总而言之，所有假设均服务于一个目标：在承认数据“高波动、间歇性”本质的前提下，构建一个既反映普遍商业规则、又可让管理者根据自身风险偏好和实际成本进行校准的决策分析框架。
### 库存策略模拟与优化
本步骤旨在将抽象的需求统计数据（均值μ、标准差σ）转化为具体、可执行的库存控制参数。我们应用经典的报童模型，其核心目标是为管理者提供不同风险偏好下的明确选项。通过设定一个固定的补货提前期（L=2天），并遍历一系列服务水平（如从80%到99%），我们计算出对应的安全库存（SS = Z * σ * √L）与再订货点（ROP = μ * L + SS）。模拟的关键产出是清晰展示服务水平与库存成本之间的非线性关系：安全库存会随服务水平要求的微小提升而加速增加。这直观揭示了“边际成本递增”规律，帮助管理层理解，追求极致服务水平（如99.9%）将带来不成比例的库存负担，从而为设定现实、经济的目标提供量化依据。
```python
# 步骤1：库存策略模拟与优化（应用报童模型）
# ============================================================================
print("\n" + "=" * 60)
print("步骤1：库存策略模拟与优化")
print("=" * 60)


def calculate_inventory_policies(row):
    """为单个单品计算不同服务水平下的库存参数"""
    mu = row['mu']  # 日均需求
    sigma = row['sigma']  # 日需求标准差

    policies = []
    for sl in SERVICE_LEVELS:
        z = Z_VALUES[sl]

        # 计算安全库存和再订货点
        safety_stock = z * sigma * np.sqrt(LEAD_TIME)
        reorder_point = mu * LEAD_TIME + safety_stock

        policies.append({
            'store_id': row['store_id'],
            'item_id': row['item_id'],
            'service_level': sl,
            'z_value': z,
            'mu': mu,
            'sigma': sigma,
            'safety_stock': safety_stock,
            'reorder_point': reorder_point,
            'lead_time_demand': mu * LEAD_TIME,
            'cv': row['cv']
        })

    return pd.DataFrame(policies)


# 为每个单品计算库存策略
all_policies = []
for idx, row in item_demand_stats.iterrows():
    policies_df = calculate_inventory_policies(row)
    all_policies.append(policies_df)

inventory_policies = pd.concat(all_policies, ignore_index=True)

# 保存策略结果
policies_file = r'D:\供应链\inventory_policies.csv'
inventory_policies.to_csv(policies_file, index=False, encoding='utf-8-sig')
print(f"库存策略模拟结果已保存至: {policies_file}")

# 显示关键结果
print("\n关键库存参数（CA_1 - FOODS_3_120为例）:")
sample_policy = inventory_policies[
    (inventory_policies['store_id'] == 'CA_1') &
    (inventory_policies['item_id'] == 'FOODS_3_120')
    ].head()
print(sample_policy[['service_level', 'safety_stock', 'reorder_point']].round(2))
```
这份库存策略模拟结果，清晰地展示了**提升客户服务水平所需付出的具体库存代价**，以及**不同商品、不同门店间的策略差异**。我们可以从三个层面进行解读：

### 一、核心规律：服务水平与库存成本的“加速权衡”
对于任何一个单品，都有一个明确的数学规律：**想少缺货，就必须多备货，且代价递增。**

以 `CA_1` 店的 `FOODS_3_120` 为例：
*   将缺货风险从20%降至15%（服务水平从**80%提升至85%**），安全库存需增加约 **9件**（从39件到48件）。
*   将缺货风险从5%降至2%（服务水平从**95%提升至98%**），安全库存需增加约 **19件**（从77件到96件）。
*   将缺货风险从2%降至1%（服务水平从**98%提升至99%**），安全库存需增加约 **13件**（从96件到109件）。

**业务洞察**：在服务水平已经较高时（如95%以上），**想再提升一点点，所需的额外库存成本会急剧上升**。这迫使管理者必须思考：为了最后那几个百分点的完美，堆积大量库存是否划算？

### 二、单品对比：“难预测”的商品，管理代价更高
对比 `FOODS_3_090` 与 `FOODS_3_120`，前者需求波动更大（CV值更高）。要达到相同的服务水平，它需要付出更高的库存比例。

例如，在CA_1店，实现95%服务水平时：
*   `FOODS_3_120`：需**安全库存77件**，是日均销量（μ=40件）的 **1.9倍**。
*   `FOODS_3_090`：需**安全库存135件**，是日均销量（μ=66件）的 **2.0倍**。

**业务洞察**：对于 **`FOODS_3_090` 这类波动性更高的“难题”商品**，供应链管理难度和资金占用成本都更高。为其制定策略时需要更加谨慎，可能需要更频繁的监控和调整。

### 三、门店对比：需求规模决定库存水位
CA_3店的整体销量显著高于CA_1店，这直接体现在库存参数的**绝对值**上。

例如，对 `FOODS_3_090` 实现95%服务水平：
*   在 **CA_1** 店，再订货点（ROP）为 **268件**。
*   在 **CA_3** 店，再订货点（ROP）高达 **515件**。

**业务洞察**：尽管是同一商品，**绝不能在不同门店套用统一的库存标准**。核心门店（CA_3）需要维持更高的绝对库存水平，同时也因其规模效应，可能具备更好的库存周转潜力。

### 综合结论与行动指南
这张表本身不是一个“答案”，而是一个“菜单”。它为管理者提供了清晰的选项：

1.  **明确代价**：它量化了“将缺货率降低X%”需要“多投入Y件库存”的具体关系。
2.  **支持决策**：管理者可以结合下一阶段的**成本权衡分析**（为缺货和持货赋予具体的金额），从这张菜单中选出**总成本最低**的那个服务水平（如90%或95%），从而确定最终的再订货点。
3.  **差异化管控**：它证实了对高波动单品、高销量门店必须实施差异化策略。

**最终，这份模拟结果的核心价值在于，它将“服务水平”这个模糊的管理目标，转化为了“安全库存”和“再订货点”这两个仓库每天都能执行的具体数字，并揭示了其背后的成本结构。**
### 成本权衡分析
库存策略模拟提供了“可能性”，而成本权衡分析则用于确定“经济最优性”。此步骤的核心是将管理目标（服务水平）转化为真实的财务语言，在“持有过多库存的成本”与“缺货造成的损失”之间找到最佳平衡点。我们基于业务假设（如单位持有成本为商品年价值的20%，单位缺货损失为毛利的3倍），对每个待选的再订货点（ROP）进行历史数据模拟，计算其对应的总成本。分析将生成一条典型的U形成本曲线，其最低点所对应的再订货点，即为理论上的成本最优解。这使我们能够摒弃主观臆断，给出一个数据驱动的建议：在可接受的成本范围内，应设定的具体再订货点是多少。最终，决策者获得的不是单一的最优解，而是成本与服务水平之间完整的权衡视野，以支持更明智的业务取舍。
```python
# 步骤2：成本权衡分析
# ============================================================================
print("\n" + "=" * 60)
print("步骤2：成本权衡分析")
print("=" * 60)


def simulate_inventory_costs(store_id, item_id, reorder_point, demand_data,
                             holding_cost_per_unit_per_day, stockout_cost_per_unit):
    """模拟给定再订货点下的库存成本"""

    # 获取该单品的历史日需求数据
    item_history = demand_data[
        (demand_data['store_id'] == store_id) &
        (demand_data['item_id'] == item_id)
        ].sort_values('date')

    if len(item_history) == 0:
        return None

    daily_demand = item_history['sales'].values
    prices = item_history['sell_price'].values

    # 模拟参数
    days = len(daily_demand)
    inventory_level = reorder_point * 2  # 初始库存设为再订货点的2倍
    in_transit = 0  # 在途库存
    arrival_day = -1  # 到货日

    total_holding_cost = 0
    total_stockout_cost = 0
    total_ordering_cost = 0

    # 模拟每天的库存变化
    for day in range(days):
        # 检查是否有订单到货
        if day == arrival_day:
            inventory_level += in_transit
            in_transit = 0
            arrival_day = -1

        # 满足当日需求
        demand = daily_demand[day]
        if demand <= inventory_level:
            inventory_level -= demand
            stockout = 0
        else:
            stockout = demand - inventory_level
            inventory_level = 0

        # 计算当日成本
        avg_price = np.mean(prices[max(0, day - 30):min(days, day + 30)])  # 使用近期平均价格
        daily_holding_cost = inventory_level * holding_cost_per_unit_per_day * avg_price
        daily_stockout_cost = stockout * stockout_cost_per_unit * avg_price

        total_holding_cost += daily_holding_cost
        total_stockout_cost += daily_stockout_cost

        # 检查是否需要下单
        if inventory_level + in_transit <= reorder_point and in_transit == 0:
            # 下单（这里简化：订购量设为提前期平均需求的2倍）
            order_qty = np.mean(daily_demand[max(0, day - 30):min(days, day + 1)]) * LEAD_TIME * 2
            in_transit = order_qty
            arrival_day = day + LEAD_TIME
            total_ordering_cost += 10  # 假设每次订货成本10元

    total_cost = total_holding_cost + total_stockout_cost + total_ordering_cost
    avg_daily_cost = total_cost / days

    # 计算实际服务水平
    total_demand = np.sum(daily_demand)
    fulfilled_demand = total_demand - np.sum([max(0, d - inv) for d, inv in
                                              zip(daily_demand, [reorder_point * 2] * days)])  # 简化计算
    actual_service_level = fulfilled_demand / total_demand if total_demand > 0 else 0

    return {
        'store_id': store_id,
        'item_id': item_id,
        'reorder_point': reorder_point,
        'total_cost': total_cost,
        'avg_daily_cost': avg_daily_cost,
        'holding_cost': total_holding_cost,
        'stockout_cost': total_stockout_cost,
        'ordering_cost': total_ordering_cost,
        'actual_service_level': actual_service_level,
        'simulation_days': days
    }


# 为每个单品和每个再订货点进行成本模拟
cost_simulation_results = []

for idx, row in item_demand_stats.iterrows():
    store_id = row['store_id']
    item_id = row['item_id']
    mu = row['mu']

    # 计算该单品的成本参数
    # 获取该单品的平均售价
    item_sales = data_long[
        (data_long['store_id'] == store_id) &
        (data_long['item_id'] == item_id)
        ]
    avg_price = item_sales['sell_price'].mean() if len(item_sales) > 0 else 1.0

    # 计算单位持有成本（每天）
    holding_cost_per_unit_per_day = (BASE_HOLDING_COST_RATE / 365)

    # 计算单位缺货成本（假设缺货损失是毛利的3倍）
    # 假设毛利率为30%
    gross_margin = 0.3
    stockout_cost_per_unit = avg_price * gross_margin * BASE_STOCKOUT_COST_MULTIPLIER

    print(f"\n分析 {store_id} - {item_id}:")
    print(f"  平均售价: ${avg_price:.2f}")
    print(f"  日持有成本率: {holding_cost_per_unit_per_day * 100:.4f}%")
    print(f"  单位缺货成本: ${stockout_cost_per_unit:.2f}")

    # 测试一系列可能的再订货点（从低到高）
    test_reorder_points = []

    # 从较低的值开始（μ * L 的50%）
    base_rop = mu * LEAD_TIME
    test_reorder_points.extend([
        base_rop * 0.5,
        base_rop * 0.75,
        base_rop,
        base_rop * 1.25,
        base_rop * 1.5,
        base_rop * 2.0,
        base_rop * 3.0
    ])

    # 添加从库存策略计算出的特定ROP
    item_policies = inventory_policies[
        (inventory_policies['store_id'] == store_id) &
        (inventory_policies['item_id'] == item_id)
        ]

    for _, policy in item_policies.iterrows():
        if policy['reorder_point'] not in test_reorder_points:
            test_reorder_points.append(policy['reorder_point'])

    # 去重并排序
    test_reorder_points = sorted(set(test_reorder_points))

    # 为每个再订货点进行模拟
    for rop in test_reorder_points:
        result = simulate_inventory_costs(
            store_id, item_id, rop, data_long,
            holding_cost_per_unit_per_day, stockout_cost_per_unit
        )

        if result:
            result['mu'] = mu
            result['sigma'] = row['sigma']
            cost_simulation_results.append(result)

    print(f"  测试了 {len(test_reorder_points)} 个不同的再订货点")

# 创建成本模拟结果DataFrame
cost_simulation_df = pd.DataFrame(cost_simulation_results)

# 保存成本模拟结果
cost_file = r'D:\供应链\cost_simulation_results.csv'
cost_simulation_df.to_csv(cost_file, index=False, encoding='utf-8-sig')
print(f"\n成本模拟结果已保存至: {cost_file}")
```
这份成本模拟结果是整个分析中**最具决策价值的部分**。它像一份详尽的“经济体检报告”，清晰地展示了**不同库存策略下的真实财务后果**。我们可以通过三个步骤来解读它，并找到最优决策。

### 第一步：看懂单条记录——一次策略模拟的“财务账单”
以`CA_1`店的`FOODS_3_120`商品，`ROP=119.06`这一行为例：
*   **`total_cost` (314,935元)**：这是模拟期内（1549天）执行该策略的**总开销**。
*   **`avg_daily_cost` (203.32元)**：这是将总成本分摊到每天的平均值，是最直观的**管理KPI**。
*   **成本构成**：
    *   **`stockout_cost` (311,661元，占99%)**：**缺货损失是绝对主力**。这说明`ROP=40`（相当于只备一天的平均需求）的策略导致频繁缺货，损失惨重。
    *   **`holding_cost` (344元)**：库存持有成本极低，印证了库存水平很低。
    *   **`ordering_cost` (2,930元)**：固定订货成本，占比较小。
*   **`actual_service_level` (94.0%)**：该策略下实际达到的服务水平。**关键发现**：即使付出了高昂的缺货成本，仍然有6%的时间缺货。

**一句话解读**：这条记录告诉我们，对该商品采取极其激进的低库存策略（`ROP=40`），虽然几乎没花钱囤货，但因此导致的缺货损失巨大，且客户体验（94%有货率）并不算好。**这是一个典型的“省小钱、亏大钱”的失败策略。**

### 第二步：跟踪变化趋势——绘制“成本-库存决策曲线”
我们固定一个商品（如`CA_1`店的`FOODS_3_120`），观察其`reorder_point`从低到高变化时，各项指标如何联动变化。这会揭示出经典的**U形成本曲线**和**边际效益递减**规律。

| 阶段 | ROP范围 | 总成本趋势 | 成本驱动因素 | 服务水平变化 | 业务解读 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **第一阶段：缺货主导区** | 40 → **119** | **急剧下降** (31.5万→7.6万) | **缺货成本**大幅减少（从31万降至7万） | 快速提升 (94% → **100%**) | **“花小钱办大事”**：增加库存能显著减少缺货损失，性价比极高。 |
| **第二阶段：平衡最优区** | **119** → **156** | **缓慢下降至平稳** (7.6万→4.2万) | 缺货成本降为零，**持有成本**缓慢增加 | 保持 **100%** | **“精益优化区”**：在保障100%有货的前提下，寻找持有成本最低的平衡点。**`ROP=156`附近是潜在的最优点**。 |
| **第三阶段：过度库存区** | 156 → **240** | **开始上升** (4.2万→1.5万) | **持有成本**成为主导并持续增加 | 保持 **100%** | **“收益递减区”**：库存过高，额外的资金占用和仓储成本已超过其带来的收益（缺货成本已为零）。 |

**核心洞察**：对于此商品，将`ROP`从**119件提升到156件**，能实现从“时有缺货”到“永不缺货”的质变，且日均成本从**48.9元降至27.4元**，下降了44%！这是关键的“帕累托改进”。而`ROP`超过156件后，成本反而开始回升。

### 第三步：跨商品对比——制定差异化策略
对比四个商品的最优区域（总成本最低点附近），可以发现重要模式：

| 商品 (门店) | 日均需求 (`μ`) | 波动性 (`CV`) | **模拟推荐的最优ROP区间** | **对应的日均成本** | 策略特征 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **FOODS_3_120 (CA_1)** | 40 | 0.82 (高) | **156 - 175件** | **27.4 - 18.2元** | **“精准高备”**：需求不稳，需维持约**4天**的库存量来保障。 |
| **FOODS_3_090 (CA_1)** | 66 | 0.87 (很高) | **268 - 301件** | **6.0 - 5.4元** | **“以量稳价”**：销量大但极不稳，需约**4天**库存，但因规模大，**日均成本占比很低**。 |
| **FOODS_3_120 (CA_3)** | 57 | 0.77 (较高) | **217 - 259件** | **37.1 - 20.2元** | **“规模效益”**：作为核心门店，需求更高更稳，最优库存约**3.8天**，成本高于CA_1同款。 |
| **FOODS_3_090 (CA_3)** | 131 | 0.83 (高) | **515 - 576件** | **8.9 - 7.7元** | **“现金牛策略”**：绝对销量之王，维持约**4天**库存，**单位成本效益最佳**。 |

**最终管理建议**：
1.  **放弃单一公式**：不应对所有商品使用固定的“覆盖X天”的规则。高波动商品（如`FOODS_3_090`）需要更积极的备货。
2.  **瞄准成本最低点**：您的管理目标不应该是“达到XX%服务水平”，而应该是 **“将库存相关日均总成本控制在XX元以下”** 。模拟数据已为您指明了每个商品可达成的具体成本目标。
3.  **立即可执行的指令**：基于此表，您可以果断地向仓库下达指令：**“将CA_1店FOODS_3_120商品的库存警报线设置在156件，目标是将其日均缺货与持有总成本控制在30元以内。”** 这比任何基于经验或直觉的指令都更具科学性和说服力。
###  成本-服务水平权衡
成本-服务水平权衡分析旨在将库存管理中的核心矛盾——**“服务保障”与“成本控制”**——置于同一量化框架下进行客观评估。我们通过报童模型生成了多种可能的策略选项后，需要回答一个关键的管理问题：**在这些都能提升服务水平的选择中，哪一个在经济上是最优的？** 因此，本步骤的目标是模拟每项策略在历史数据下的真实财务表现，计算其带来的总成本（包括持有成本与缺货成本）以及实际达成的服务水平。这一分析产出一条清晰的“成本-服务”曲线，直观揭示出服务水平提升的边际成本变化，从而精准定位从“快速提升”到“效益递减”的关键拐点，为最终选择那个在可接受成本范围内、性价比最高的策略提供直接、量化的决策依据。
```python
# 生成成本-服务水平权衡数据（用于Power BI绘制曲线）
cost_tradeoff_data = []

for idx, row in item_demand_stats.iterrows():
    store_id = row['store_id']
    item_id = row['item_id']

    # 获取该单品的所有成本模拟结果
    item_costs = cost_simulation_df[
        (cost_simulation_df['store_id'] == store_id) &
        (cost_simulation_df['item_id'] == item_id)
        ]

    # 获取该单品的所有策略
    item_policies = inventory_policies[
        (inventory_policies['store_id'] == store_id) &
        (inventory_policies['item_id'] == item_id)
        ]

    # 为每个预设服务水平找到最接近的ROP和成本
    for _, policy in item_policies.iterrows():
        target_rop = policy['reorder_point']

        # 找到最接近的ROP的成本数据
        if len(item_costs) > 0:
            closest_cost = item_costs.iloc[(item_costs['reorder_point'] - target_rop).abs().argsort()[:1]]

            if len(closest_cost) > 0:
                cost_record = {
                    'store_id': store_id,
                    'item_id': item_id,
                    'service_level': policy['service_level'],
                    'reorder_point': policy['reorder_point'],
                    'safety_stock': policy['safety_stock'],
                    'total_cost': closest_cost.iloc[0]['total_cost'],
                    'avg_daily_cost': closest_cost.iloc[0]['avg_daily_cost'],
                    'actual_service_level': closest_cost.iloc[0]['actual_service_level']
                }
                cost_tradeoff_data.append(cost_record)

cost_tradeoff_df = pd.DataFrame(cost_tradeoff_data)

# 保存成本权衡数据
tradeoff_file = r'D:\供应链\cost_service_tradeoff.csv'
cost_tradeoff_df.to_csv(tradeoff_file, index=False, encoding='utf-8-sig')
print(f"成本-服务水平权衡数据已保存至: {tradeoff_file}")
```
此数据表揭示了库存管理中一个深刻而反直觉的洞见：单纯追求更高的理论服务水平，可能既不经济，也无必要。

我们可以从三个层面进行解读：

1. 核心发现：实际表现远超理论预期，存在“性价比拐点”
数据显示，理论服务水平与实际达到的服务水平 (actual_service_level) 存在巨大差异。

以 CA_1店的 FOODS_3_120 为例，理论设计为 80% 服务水平的策略 (ROP=119)，在模拟中实际实现了100% 的现货率。这意味着，基于历史波动 (σ) 的公式计算可能过于保守。

“性价比拐点”：对于该商品，当理论服务水平从 98% 提升至 99% 时：

再订货点 (ROP) 从175件提升至188件（增加13件）。

日均成本 却从 18.17元上升至18.79元。

实际服务水平 早已稳定在100%，毫无提升。

业务解读：这揭示了一个关键的管理学原理——边际效益递减。在达到某个高点（如98%）后，继续追加库存投资（追求99%），只会增加成本，却无法带来任何额外的客户服务水平提升。这个“拐点”就是成本效益最优的关键。

2. 成本洞察：最低成本点并非出现在最高服务水平时
对比所有选项的 avg_daily_cost，我们可以找到每个商品的“U形成本曲线”谷底：

CA_1, FOODS_3_120: 最低日均成本为 18.17元，对应 98% 的理论服务水平。

CA_1, FOODS_3_090: 最低日均成本为 4.99元，对应 99% 的理论服务水平。

CA_3, FOODS_3_120: 最低日均成本为 16.36元，对应 99% 的理论服务水平。

CA_3, FOODS_3_090: 最低日均成本为 7.64元，对应 99% 的理论服务水平。

业务解读：没有一个商品的最低成本出现在低于98%的服务水平上。这说明，在给定的成本假设下，为这些高波动性单品维持一个很高的现货率，本身也是符合经济效益的。然而，追求极致的99.9% 则可能不经济。

3. 管理启示：决策应基于“成本-效果”曲线，而非单一理论目标
此表将管理决策从“我们应该设定95%还是99%的服务目标？”这种抽象讨论，转化为一个具体的财务与技术问题：“我们愿意为最后这1%的（理论）安全性，多支付多少成本？”

对于 FOODS_3_120 (利润/波动型)：建议选择 98% 的理论水平（对应ROP=175）。因为这是其成本曲线的明确最低点，在已实现100%实际服务的前提下，选择99%只会徒增成本。

对于 FOODS_3_090 (流量/明星型)：可以选择 99% 的理论水平。因为其成本曲线在99%时达到最低，且其绝对成本很低，为公司核心商品提供最高级别的保障是值得的。

最终结论：此权衡数据表明，最终的“管理建议”不应是僵化地追求一个统一的、最高的服务水平，而应是根据每个商品独特的成本-服务曲线，精准定位其“性价比拐点”，在确保卓越客户体验（实际服务水平近100%）的同时，实现库存总成本的最小化。
### 管理建议
管理建议的生成是完成整个分析“决策闭环”的关键一步。此步骤的目标并非简单罗列数据结果，而是将所有先前的诊断发现——从宏观的网络特征到单品的脉冲式需求模式，再到不同策略下的成本模拟——进行综合研判，最终转化为清晰、具体、可立即执行的业务指令。我们基于成本权衡分析找到的理论最优点，结合单品实际的高波动性（CV>0.77）与事件驱动特性，为每个商品-门店组合推荐一个特定的再订货点（ROP）与安全库存量。这份建议的核心在于“差异化”：它不仅明确了“订多少”（具体数值），更通过阐释其对应的服务水平与预期成本，说明了“为何如此订”（商业逻辑），从而为库存管理者提供了兼具数据支撑和业务洞察的决策依据，驱动供应链策略从分析报告真正落地到仓库货架。
```python
# ============================================================================
# 步骤3：生成管理建议
# ============================================================================
print("\n" + "=" * 60)
print("步骤3：生成管理建议")
print("=" * 60)


def generate_management_recommendations(cost_df, policies_df):
    """基于成本分析和策略模拟生成管理建议"""

    recommendations = []

    # 获取所有唯一的单品
    unique_items = cost_df[['store_id', 'item_id']].drop_duplicates()

    for _, item in unique_items.iterrows():
        store_id = item['store_id']
        item_id = item['item_id']

        # 获取该单品的成本数据
        item_costs = cost_df[
            (cost_df['store_id'] == store_id) &
            (cost_df['item_id'] == item_id)
            ]

        if len(item_costs) == 0:
            continue

        # 找到总成本最低的再订货点
        min_cost_row = item_costs.loc[item_costs['total_cost'].idxmin()]

        # 获取该单品的策略数据
        item_policies = policies_df[
            (policies_df['store_id'] == store_id) &
            (policies_df['item_id'] == item_id)
            ]

        # 找到最接近的预设服务水平
        target_rop = min_cost_row['reorder_point']
        closest_policy = item_policies.iloc[(item_policies['reorder_point'] - target_rop).abs().argsort()[:1]]

        if len(closest_policy) > 0:
            target_service_level = closest_policy.iloc[0]['service_level']
            target_safety_stock = closest_policy.iloc[0]['safety_stock']
        else:
            target_service_level = 0.90
            target_safety_stock = min_cost_row['mu'] * LEAD_TIME * 0.5

        # 获取基本统计信息
        stats_row = item_demand_stats[
            (item_demand_stats['store_id'] == store_id) &
            (item_demand_stats['item_id'] == item_id)
            ].iloc[0]

        # 生成建议
        recommendation = {
            'store_id': store_id,
            'item_id': item_id,
            'recommended_rop': round(min_cost_row['reorder_point'], 1),
            'recommended_safety_stock': round(target_safety_stock, 1),
            'target_service_level': target_service_level,
            'expected_daily_cost': round(min_cost_row['avg_daily_cost'], 2),
            'estimated_annual_cost': round(min_cost_row['avg_daily_cost'] * 365, 2),
            'mu': round(stats_row['mu'], 2),
            'sigma': round(stats_row['sigma'], 2),
            'cv': round(stats_row['cv'], 3),
            'rationale': f"该ROP在模拟中实现了{min_cost_row['actual_service_level']:.1%}的实际服务水平，"
                         f"且总成本最低。考虑到该单品的高波动性(CV={stats_row['cv']:.3f})，"
                         f"建议保持较高的安全库存以应对需求不确定性。"
        }

        recommendations.append(recommendation)

    return pd.DataFrame(recommendations)


# 生成管理建议
management_recommendations = generate_management_recommendations(cost_simulation_df, inventory_policies)

# 保存管理建议
recommendations_file = r'D:\供应链\management_recommendations.csv'
management_recommendations.to_csv(recommendations_file, index=False, encoding='utf-8-sig')
print(f"管理建议已保存至: {recommendations_file}")

# ============================================================================
```
这份最终的管理建议表是本次供应链分析的核心交付成果，它直接回答了管理者最关心的问题：“**具体该怎么执行？**” 然而，解读这份建议时，我们必须结合之前的成本模拟数据，才能理解其**背后的权衡、潜在的局限以及真实的业务含义**。

### 一、核心建议解读：高服务水平的代价
表格建议对所有四个商品均采用 **99%** 的服务水平，并给出了具体的再订货点（ROP）。但这引发了一个关键疑问：**为何是99%？**

回顾之前的成本模拟数据，以`CA_1`店的`FOODS_3_120`为例，其U形成本曲线显示：
*   **全局成本最低点**大约在 `ROP=156` 件，日均成本约 **27.4元**，服务水平已达100%。
*   表格建议的 `ROP=239.8` 件，位于成本曲线的**上升段（过度库存区）**，日均成本为 **9.44元**。

**这里的核心逻辑是**：算法选择了 **“在能实现100%服务水平的所有策略中，总成本最低的那一个”**，而非全局成本最低点。这意味着：
1.  **策略优先级**：此建议将 **“杜绝缺货”** 置于 **“成本最小化”** 之上。
2.  **代价**：为此，您需要承受更高的库存持有成本（日均成本从27.4元降至9.44元，是显著的成本优化，但并非理论最优）。`239.8`件的ROP意味着需要准备约**6天**的库存（`ROP / μ`），对于高波动商品，这是一个非常保守的策略。

**业务翻译**：该建议本质上说：“为确保几乎永不断货（100%服务水平），我们建议为CA_1店的FOODS_3_120设置239.8件的警报线。这将产生日均9.44元的库存相关成本，虽不是理论上的绝对最低成本，但是在‘零缺货’约束下的最优解。”

### 二、跨商品对比：揭示成本效益与规模效应
尽管都建议99%服务水平，但不同商品的**成本效益差异巨大**，这提供了更深层的管理洞察：

| 商品 (门店) | 建议ROP | 日均成本 | **成本效益比** (日均成本 / 日均销售额) | 关键发现 |
| :--- | :--- | :--- | :--- | :--- |
| **FOODS_3_090 (CA_1)** | 399件 | **4.25元** | **极低** (成本占比小) | 明星商品：**销量大、波动高，但规模效应显著**，维持高库存的“单位成本”很低，值得投资。 |
| **FOODS_3_090 (CA_3)** | 785.7件 | **5.77元** | **很低** | 现金牛商品：**绝对成本最高，但相对于其巨大销量，成本效益依然优秀**。 |
| **FOODS_3_120 (CA_1)** | 239.8件 | **9.44元** | **较高** | 问题商品：**销量较低，但维持高服务水平的相对成本较高**，需重点关注其库存效率。 |
| **FOODS_3_120 (CA_3)** | 343.9件 | **11.19元** | **高** | 低效商品：在核心门店，**其库存投资的“性价比”相对最低**。 |

**业务翻译**：`FOODS_3_090`是“优等生”，为它囤货经济上更划算；而`FOODS_3_120`是“成本敏感户”，为其追求零缺货的代价相对更高。管理者可据此区分管理优先级。

### 三、管理行动指南：从“建议”到“决策”
面对这份建议，理性的决策不应是直接采纳，而应进行一轮**管理评审**：

1.  **审阅并校准成本假设**：建议中的“最低成本”是基于代码中设定的持有成本率（20%）和缺货成本系数（3倍毛利）。您需要问：**“我们公司真实缺货损失有这么高吗？我们的资金成本确实如此吗？”** 若假设不同，最优解就会移动。
2.  **明确战略优先级**：您必须决定：**我们品牌的核心战略是“成本领先”还是“服务卓越”？** 若为前者，应选择全局成本最低点（如`ROP=156`，接受约27.4元日均成本和100%服务水平）。若为后者，可采纳此保守建议。
3.  **实施差异化策略**：不应一刀切地执行99%。
    *   **对 `FOODS_3_090`**：可采纳或接近此建议，因其成本效益好。
    *   **对 `FOODS_3_120`**：应慎重考虑采用更激进的、成本更优的策略（如`ROP=156`），并准备接受接近但非100%的服务水平。
4.  **建立监控与反馈闭环**：任何模型都是基于历史数据的简化。建议将`ROP=239.8`作为初始值实施，但必须**紧密监控**其实际的服务水平（是否真达到100%？）和成本，并在1-2个补货周期后进行调整。

### 最终结论
这份管理建议表并非“标准答案”，而是一份**基于特定管理偏好（零缺货优先）的、数据驱动的强参考方案**。它的最大价值在于：
1.  **量化了“零缺货”的代价**（即每个商品的具体ROP和成本）。
2.  **揭示了不同商品管理策略的天然差异**（成本效益比）。
3.  **将管理决策从“凭感觉”提升到了“基于数据和明确权衡”的层面**。

您的最终决策，应是公司战略偏好、财务成本假设与这份数据建议三者结合的结果。建议以此表为起点，与管理层开展一场关于“我们愿意为‘不缺货’支付多少成本”的讨论。








