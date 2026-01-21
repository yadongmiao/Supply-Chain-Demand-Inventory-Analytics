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
### 核心结论：四象限门店画像

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
### 下一阶段分析对象选定

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








### 供应链管理策略建议

基于门店画像与网络特性，提出以下差异化策略：

1.  **门店级库存策略**：
    *   **CA_3 (现金牛)**：实施**高效自动化补货**。可采用较低的**安全库存系数**和固定的**再订货点（ROP）**，目标是最大化周转率与成本效率，并将其策略作为网络标杆。
    *   **CA_1 (问题明星)**：实行**重点监控与动态管理**。必须配置更高的**安全库存**，并采用更短周期的预测回顾（如每周），可考虑引入更高级的预测模型（如结合促销事件）。此门店是提升整体网络服务水平的关键。
    *   **CA_2 (风险门店)**：采取**保守的成本控制策略**。优先考虑按订单补货或设置极低的安全库存，以规避库存积压风险，接受较低的服务水平。
    *   **CA_4 (长尾门店)**：推行**简化管理**。采用周期性的集中复查与补货（如每两周一次），减少日常管理投入。

2.  **网络级运营优化**：
    *   在补货计划中，应根据门店的**CV值（波动性）** 设定差异化的库存覆盖天数，而非“一刀切”。
    *   在物流配送上，可对CA_1和CA_3采用更频繁的配送班次，以快速响应需求变化并降低平均库存水平。

