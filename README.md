# Supply-Chain-Demand-Inventory-Analytics
## 数据描述
本数据集为沃尔玛销售预测项目提供了一套完整、多层级的历史数据，旨在支持从宏观战略分析到微观库存决策的全方位供应链洞察。数据核心为覆盖美国三州（CA、TX、WI）门店的长周期日度销售记录（d_1 至 d_1913），并通过关联表整合了时间、价格与产品信息。  

数据采用典型的星型结构：主事实表记录了“产品-门店-日期”粒度的销量；calendar.csv提供日期维度的星期、假日及特殊事件属性；sell_prices.csv则记录了动态的产品定价信息。产品维度按商品、类别、部门进行了层级划分，门店维度也归属于各州。  

该数据集的核心价值在于其丰富的分析维度与层次。在时间上，超过5年的日度数据可支持趋势分解与季节性建模；在空间上，州、门店的划分便于进行区域对比与网络聚合分析；在产品上，从宽泛的品类（如FOODS）到具体单品（item_id）的层级结构，完美契合“从面到点”的分析漏斗。同时，价格、促销等解释变量的存在，使得需求驱动因素的量化分析成为可能，为精准的需求预测和后续的库存策略模拟（如计算安全库存、再订货点）提供了坚实基础。  
## 分析目标
本次分析旨在运用沃尔玛多层次销售数据，解决供应链管理中“如何将历史销售数据转化为精准、可执行的库存决策”这一核心问题。我们将通过一个“从面到点”的三层漏斗式分析框架，首先识别出核心销售区域和销售规律，进而剖析该区域内供应链网络的聚合效应与门店需求特征，最终聚焦于代表性单品与门店，量化其需求规律，并应用库存模型模拟不同策略下的成本与服务权衡，从而为特定商品制定出数据驱动的、最优的安全库存与再订货点建议，实现需求预测到库存策略的闭环落地。
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
## 分析计划
为实现“从历史数据到库存决策”的分析目标，我设计并执行了一套“从面到点”的三层漏斗式分析框架。该框架遵循“宏观定位 → 中观诊断 → 微观优化”的逻辑，逐层收敛分析焦点，最终形成可执行的策略建议。

### **第一阶段：品类宏观分析（面向全网络）**
**核心目标**：评估FOODS品类的整体市场表现与战略重要性，为资源聚焦提供依据。
**关键问题**：
1.  该品类的全国销售呈现何种长期趋势与季节性规律？
2.  销售在地理区域（CA, TX, WI）是否存在绝对主导的核心市场？
3.  从宏观层面看，该品类的整体需求是稳定的还是波动剧烈的？

| 步骤 | 分析动作与方法 | 预期交付物（分析前定义） |
| :--- | :--- | :--- |
| 1. **趋势分析** | 聚合全国月度销售额，绘制时间序列折线图，观察长期趋势与周期性。 | 全国销售趋势图表，揭示增长态势与季节性峰谷。 |
| 2. **区域对比分析** | 计算各州销售占比，并对比各州的月度销售趋势曲线。 | 1. 销售区域占比图（如饼图）。<br>2. 分州销售趋势对比图。<br>3. **核心市场判定**：基于份额与趋势，确定下一阶段的重点分析区域。 |
| 3. **稳定性初探** | 计算全国月度销售数据的变异系数（CV），量化整体波动水平。 | 品类宏观需求波动性评估（高/中/低），为后续网络分析设定波动性基准。 |

**本阶段输出与决策**：明确FOODS品类的宏观画像与核心市场，**正式锁定一个州** 作为第二阶段深度分析的战场。

### **第二阶段：品类区域分析（聚焦CA网络）**
**核心目标**：深入理解品类在核心市场供应链网络中的运行特征，识别差异化管理的依据。
**关键问题**：
1.  **网络结构**：集中式库存管理（中央仓）是否有效平滑了终端（门店）需求的波动？
2.  **节点画像**：网络内各门店的需求模式有何不同？如何对它们进行分类？
3.  **网络协同**：各门店的需求波动是相互独立的，还是同步变化的？这对集中库存策略有何影响？

| 步骤 | 分析动作与方法 | 预期交付物（分析前定义） |
| :--- | :--- | :--- |
| 1. **网络聚合效应验证** | 分别计算“门店级平均需求波动（CV）”与“中央仓级总需求波动（CV）”，并进行对比。 | 量化聚合效应大小的关键对比指标（如波动降低百分比）及结论。 |
| 2. **门店需求画像构建** | 计算各门店的核心指标（销售额、日均销量、需求CV），并运用**ABC分类法**与**四象限分析法**（销量 vs 波动性）进行分层。 | 1. 门店多维度画像表与散点分布图。<br>2. 门店分类结果（如：现金牛、问题明星等）。<br>3. **代表性门店筛选**：确定1-2家最具分析价值的门店进入下一阶段。 |
| 3. **门店需求相关性分析** | 计算各门店日销售额时间序列之间的相关系数矩阵。 | 门店间需求同步性评估报告，用于解释网络聚合效应的实际效果，并为网络级策略提供依据。 |

**本阶段输出与决策**：获得CA州供应链网络的详细诊断报告，**筛选出2家需求模式最具代表性的门店**，作为单品级深度优化的对象。

### **第三阶段：单品微观诊断（聚焦特定门店与单品）**
**核心目标**：为具体商品制定数据驱动的、最优的库存控制参数（安全库存、再订货点），完成分析闭环。
**关键问题**：
1.  **需求规律**：目标单品在特定门店的历史需求服从何种统计分布？其关键参数（均值、标准差）是多少？
2.  **策略模拟**：根据其需求规律，应用库存模型，不同服务水平对应何种库存参数？
3.  **经济最优**：在“持有库存的成本”与“缺货的损失”之间，如何找到总成本最低的平衡点？
4.  **落地建议**：最终应设定怎样的具体参数？实施时需注意什么？

| 步骤 | 分析动作与方法 | 预期交付物（分析前定义） |
| :--- | :--- | :--- |
| 1. **核心单品筛选** | 在目标门店内，依据帕累托法则（80/20法则）按销售额筛选TOP单品。 | 用于深度分析的目标单品清单。 |
| 2. **单品需求深度量化** | 计算单品历史日需求的详细统计量（均值、标准差、分位数、零销天数等），并绘制直方图与理论分布曲线进行拟合优度检验。 | 1. 单品需求核心参数表。<br>2. 需求分布形态诊断结论（是否符合经典模型假设）。 |
| 3. **库存策略参数化模拟** | 设定业务参数（提前期、目标服务水平范围、成本假设），应用报童模型计算不同服务水平下的安全库存(SS)与再订货点(ROP)。 | 库存策略模拟结果表，展示服务水平-库存成本的权衡关系。 |
| 4. **成本权衡分析与优化** | 基于历史数据模拟不同ROP策略下的总成本（持有成本+缺货成本），绘制成本曲线，寻找成本最低点。 | 1. U形成本曲线图。<br>2. 经济最优再订货点(ROP)建议及对应的预期成本与服务表现。 |
| 5. **管理建议生成** | 综合以上所有分析，生成包含具体ROP、安全库存、目标成本及实施指导的差异化建议。 | 可立即用于试点执行的、数据驱动的库存管理指令清单与校准框架。 |

**本阶段最终交付**：形成针对具体商品-门店组合的、量化且差异化的库存策略方案，并建立从数据分析到策略执行、再到反馈校准的完整管理闭环。
## 1.品类宏观分析
### 1.1趋势与区域分析
在趋势分析中，我们将日度销售数据按月份聚合，绘制全国月度销售曲线。该分析旨在清晰地揭示品类的长期增长态势与周期性规律。在区域对比中，我们进一步将销售数据按州（CA, TX, WI）与月份进行分组，通过多系列折线图直观对比不同市场的需求走势，并辅以销售占比饼图量化各州的市场地位。 
<img width="1168" height="660" alt="QQ_1768875141812" src="https://github.com/user-attachments/assets/6cee9abd-fa9e-469e-bc8b-6635828c6c4a" />
根据对 FOODS 品类的宏观分析，我们获得了以下核心洞察：从全国整体趋势来看，该品类销售呈现波动性增长的态势，表明市场总需求在周期性起伏中持续扩张。其需求呈现出显著的季节性规律，每年以7、8、9月为销售峰顶，这很可能与夏季假期、户外活动增加等消费旺季紧密相关。在地域分布上，加利福尼亚州（CA） 无疑是该品类的绝对核心市场，其销售额持续占据全国总量的 42.57%，显著领先于其他各州。这三大结论共同指明，加州不仅是 FOODS 品类的基本盘，也是其季节性增长动能的主要贡献者，因此将其作为后续供应链网络与单品深度分析的重点区域具有充分的战略依据。
### 1.2稳定性分析
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
根据对FOODS品类全国月度销售数据的量化分析，其需求呈现出高度的稳定性。在总计64个月的观察期内，月度平均销售量约为70.5万单位，其波动（标准差约为11.8万）相对有限。关键的衡量指标——变异系数（CV）仅为0.167，远低于0.3的经验阈值。这一结果表明，FOODS品类作为一个整体的需求模式可预测性较强。稳定的需求为供应链管理创造了有利条件：基于历史数据的预测模型将表现出更高的可靠性，这使得我们能够做出更为精准的销售预估。在此基础之上，由于整体需求的不确定性较低，在制定全国层面的库存计划时，为应对波动而预留的安全库存水平可以相对保守，从而有助于提升资金使用效率。这种宏观层面的可预测性，也为后续我们聚焦于核心区域（CA州），开展更为精细的网络与单品层面优化，提供了一个波动性较小的、稳健的分析起点。
### 1.3结论和建议
**结论**：于以上对FOODS品类的宏观分析，我们得出以下综合性结论：该品类在全美市场展现出稳健的基本盘特征，其需求在周期性波动中保持长期增长趋势，并呈现出以夏季（7-9月）为顶峰的显著季节性规律。从地域结构看，市场集中度较高，加利福尼亚州（CA）以超过四成的销售占比确立了其不可动摇的核心市场地位，是驱动全国销售表现的关键引擎。更为重要的是，全国层面的需求波动性较低（变异系数CV为0.167），这表明品类整体需求模式相对稳定且可预测，为实施精细化供应链管理奠定了良好的基础。

**建议**：据此，我们提出如下管理建议：公司供应链战略应明确将资源向加利福尼亚州（CA）进行倾斜，将其定位为FOODS品类供应链优化与模式探索的“战略试验区”。在运营层面，应充分利用需求整体稳定的特点，建立更具效率的全国性库存计划基准；同时，必须前瞻性地为每年第三季度的季节性销售高峰制定专项的产能、物流与库存预案，以捕获增长机遇并规避断货风险。此宏观层面的判断与决策，将为我们后续深入CA州内部供应链网络的具体诊断，提供清晰的方向和焦点。
## 2.品类区域分析（聚焦CA网络）
### 2.1网络聚合效应验证
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
### 2.2门店需求画像分析
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

# 7. 代表性门店筛选
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
分析发现，加州四家门店在“销售规模-需求稳定性”维度上差异显著，可被清晰划分至四个管理象限，其特征与业务定位如下表所示：

| 门店ID | 象限定位 | 日均销量(件) vs 中位数 | 需求波动(CV) vs 中位数 | ABC类别 | 业务画像与核心特征 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **CA_3** | **第二象限：高销量、低波动** | 3,929 **≫** 2,316 | 0.205 **<** 0.227 | A类 | **“现金牛门店”**：销量最高且需求最稳定，是网络的利润基石与效率标杆，可预测性极强。 |
| **CA_1** | **第一象限：高销量、高波动** | 2,813 **>** 2,316 | 0.249 **>** 0.227 | A类 | **“问题明星门店”**：销售额贡献第二，但需求波动性突出，是库存管理与预测优化的关键挑战点。 |
| **CA_2** | **第三象限：低销量、高波动** | 1,819 **<** 2,316 | 0.317 **≫** 0.227 | B类 | **“风险门店”**：销量有限且需求极不稳定，预测误差大，属于高管理成本、低收益的类型。 |
| **CA_4** | **第四象限：低销量、低波动** | 1,474 **<** 2,316 | 0.194 **<** 0.227 | C类 | **“长尾门店”**：销量小但需求平稳，运营简单，管理成本低。 |

该画像表明，加州的供应链网络并非同质化，而是由需求模式迥异的门店组成。若采用统一的库存策略，将无法匹配各店实际风险与需求特点，必然导致在CA_1、CA_2等店面临高缺货或高库存风险，而在CA_3、CA_4等店则可能存在资源浪费。因此，**差异化、精细化的库存策略是管理该网络的核心**。

### 2.3门店相关性分析
门店相关性分析的具体方法是将各门店的日销售额数据构建为时间序列，并计算它们两两之间的皮尔逊相关系数，以此量化不同门店需求波动在时间上的同步程度。该分析旨在从数据上验证一个关键假设：如果各门店需求受相同的外部因素（如全州范围的促销、节假日或天气变化）驱动而呈现同涨同跌的趋势，那么将它们的需求简单加总就无法通过“风险共担”来有效平滑波动，反而可能叠加放大整体不确定性，这直接关系到集中式库存策略的有效性评估与后续差异化策略的制定。
```python
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
```

通过计算门店需求间的相关性，我们发现了影响供应链网络效率的一个关键特性：门店间日销售额的相关系数矩阵显示，网络中存在强烈的**需求同步性**。特别是CA_1、CA_3、CA_4三店之间，相关系数均高于0.84，全部门店的平均相关系数达**0.75**，属于高度正相关。这一发现直接解释了为何经典的“风险共担”（集中库存以平滑波动）效应在本网络中表现不显，即聚合后中央仓的CV（0.212）比门店平均CV（0.241）有所降低（得益于规模基数增大），但降低幅度（12.1%）远低于理想中独立需求下的理论值。甚至可能出现“聚合后波动不减反增”（标准差）的反直觉现象。其根本原因在于，当所有门店的需求受相同因素（如全州促销、节假日、天气）驱动而**同涨同跌**时，在中央仓层面汇总需求无法有效抵消波动。相反，波动会被叠加并放大。对于需求高度正相关的网络，简单地集中库存至中央仓所带来的效益有限。供应链弹性的提升应侧重于**建立更灵活的响应机制**（如敏捷的运输能力）和**制定基于门店画像的差异化库存参数**，而非完全依赖集中库存来降低安全库存水平。

### 2.4结论和建议
**结论**：综合网络聚合效应验证、门店需求画像及门店相关性分析，本阶段研究揭示了FOODS品类在加州供应链网络中的核心特征与关键挑战。量化分析证实了集中库存能够平滑需求波动，但门店间高度正相关的需求同步性（平均相关系数达0.75）显著削弱了经典的“风险共担”效应，使得中央仓层面的波动降低幅度有限。更关键的是，门店画像分析显示网络内部存在显著的异质性，四家门店依据销量与波动性被清晰地划分为“现金牛”、“问题明星”、“风险门店”和“长尾门店”四种截然不同的业务类型。这一系列发现表明，加州的供应链网络并非一个同质化整体，而是一个由需求模式、风险等级和战略价值各异的节点构成的复杂系统。若继续沿用统一的库存与补货策略，将不可避免地导致在高波动门店面临缺货风险，而在稳定门店造成资源闲置，从而无法实现网络整体效率的最优。  
**建议**：基于上述结论，我们建议供应链管理策略必须从“一刀切”模式转向“精准化、差异化”模式。在门店层面，应依据其画像实施定制化策略：对“现金牛”门店CA_3推行高效自动化补货，以较低的安全库存系数最大化周转效率；对“问题明星”门店CA_1则需实施重点监控与动态管理，配置更高的安全库存并采用更短周期的预测回顾，以管控其突出的波动风险；对“风险门店”CA_2采取保守的成本控制策略，可设置极低的安全库存以规避积压；对“长尾门店”CA_4则采用简化的周期性复查管理。在网络运营层面，应建立与门店波动性（CV值）挂钩的差异化库存覆盖天数标准，并对CA_1、CA_3等高价值或高波动门店部署更敏捷的物流响应，如增加配送频次，以快速应对需求变化，从而在整体上构建一个兼具韧性、效率与成本优势的供应链网络。    
**下一阶段分析对象选定**：为将区域网络的分析结论进一步落地，并为库存决策提供可执行的微观依据，下一阶段的分析将聚焦于单品层面。我们选定CA_1与CA_3两家门店作为深入诊断的对象。选择CA_1是因为它作为“高销量、高波动”的典型代表，是当前网络中最棘手的管理挑战，对其的优化能直接解决供应链风险的核心痛点。同时选择CA_3，则是因为它作为“高销量、低波动”的效率标杆，能够确立成本最优策略的基准。通过对这两家特征迥异的门店进行单品级深度分析，我们将能够形成一套覆盖从“应对高风险”到“追求高效率”的完整策略图谱，从而完成从宏观战略到微观执行的完整决策闭环。


## 3.单品微观诊断（聚焦CA区域，特定门店与单品）
### 3.1单品筛选与数据准备
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
**核心单品高度集中**：我们共筛选出两个独特的单品：FOODS_3_090 与 FOODS_3_120。它们同时是CA_1和CA_3两家门店销售额排名前二的商品。这一结果在数据层面印证了第二阶段关于“门店需求高度相关”的结论，表明两家主力门店不仅整体需求模式同步，其核心销售引擎也完全一致。
**单品业务画像初显**：对比两个单品发现，FOODS_3_120 呈现出“高单价、低流量”的特征（总销售额高，但总销量低），可归类为 “利润型”商品；而 FOODS_3_090 则呈现“低单价、高流量”的特征（总销量和日均销量极高），属于 “流量型”商品。这一根本差异预示着，后续的库存策略必须进行差异化设计：前者应更关注利润保障与损耗控制，后者则更需平衡缺货损失与高周转带来的库存持有成本。
### 3.2单品需求深度分析
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

```
基于对 `FOODS_3_090` 与 `FOODS_3_120` 在两个门店的需求统计深度量化，我们获得了对其需求模式的精确画像。在需求水平上，`FOODS_3_090` 的日均需求量显著更高，尤其在CA_3门店达到130.9件，是其作为“流量型”商品的核心体现，而CA_3门店各单品销量全面领先也印证了其“高销量”定位；各单品需求中位数与均值接近，表明数据未受极端值严重扭曲，仅呈轻微右偏。在波动性方面，分析揭示了巨大挑战：所有单品日需求的标准差几乎与其均值持平，而变异系数（CV）均处于0.77至0.87的高位，远超0.5的阈值，这标志着需求的相对波动性极高，可预测性差，经典库存模型的稳定假设在此面临根本性挑战。进一步观察需求分布，巨大的四分位距和极端值（存在零销量日与数倍于均值的峰值销量）直观揭示了需求的极度离散与“脉冲式”特征。最值得警惕的是零销售模式，所有单品均有18.7%至24.5%的日子无销售，且最长连续零销售天数高达116至229天，这并非偶然缺货，而是一种常态化的间歇性模式，暗示商品可能受促销或特定场合驱动，若在淡季误堆积库存将导致严重的资金冻结风险。综合来看，CA_3店在销量与稳定性上均优于CA_1店，巩固了其“高效现金牛”画像；而 `FOODS_3_090` 作为规模更大、波动也更大的单品，管理复杂度最高。核心结论是，这两个贡献主要销售额的“核心畅销品”，其需求本质是**高频、高波动、间歇性、脉冲式**的，而非稳定流动的常青商品，这对制定其库存策略提出了截然不同的要求。
### 3.3需求分布可视化诊断
在量化了单品需求的核心统计参数后，我们进一步通过可视化手段直观探查其需求分布的真实形态。本步骤旨在将抽象的统计指标（如高CV值、高零销比例）转化为具体的图形证据，核心目标是检验历史需求数据是否符合经典库存模型（如报童模型）所依赖的正态分布假设。我们通过生成每个单品的需求直方图，并叠加其基于历史均值和标准差拟合的正态分布曲线，来直接观察实际分布与理论模型的偏离程度。这一分析至关重要，因为它决定了后续库存策略模拟的根基是否可靠：若拟合良好，则可直接应用经典公式；若显著偏离，则必须调整策略框架，以匹配真实的、脉冲式或间歇性的需求模式。
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
可视化结果清晰揭示了一个关键事实：所有单品的历史需求直方图与基于其自身参数拟合的正态分布曲线均存在显著偏离。图形直观地展现了此前量化分析所暗示的“零值堆积”（在零销量处有异常高的柱状）和“右偏长尾”（少数极高销量形成的拖尾）现象。这一结果是对前序量化诊断的终极视觉确认，它表明，FOODS_3_090与FOODS_3_120的需求并非围绕均值平稳波动，而是呈现出一种由大量零销日与偶发爆发日构成的“间歇性脉冲”模式。因此，直接套用基于正态假设的经典库存模型来计算安全库存与再订货点将严重脱离实际，可能导致在平时库存过剩，而在促销或旺季时又迅速缺货。

鉴于此，后续的库存策略优化必须建立在一个增强的框架之上。我们提议构建一个 “基线+事件”的双轨自适应库存系统。该系统的核心思想是：首先，仍可使用经典模型为相对平稳的“基线需求”计算出一个基准库存水平；但更重要的是，必须与营销日历深度集成，主动识别并预测那些导致需求脉冲的“极端日期”（如促销、节假日）。在这些事件来临前，系统应动态触发预案，将库存水位临时切换至更高的“事件安全库存”水平。这一策略的本质，是将供应链管理从被动应对不可预测的波动，升级为主动规划和响应已知的业务节奏。
### 3.4参数设置
在参数与假设设定阶段，我们严格遵循“基于业务逻辑、适配数据特性、支持管理决策”的原则。所有关键参数均非随意指定，其背后是行业经验与数据特征的结合：

补货提前期 (L=2天)：采用零售杂货行业对高频补货的常规假设。这是一个基准值，管理者可根据实际物流效率进行替换，模型将自动重新计算所有库存参数。

服务水平范围与Z值 (80% 至 99%)：选择这一范围旨在覆盖从“成本优先”到“服务优先”的完整管理策略光谱。特别地，我们纳入了95%以上区间的分析，因为对于CV值极高的单品，服务水平每提升1个百分点，所要求的安全库存增量将急剧扩大，这能为管理层揭示边际效益的拐点。

成本参数的核心逻辑：

持有成本率 (年化20%)：此为零售库存的典型综合成本，包含资金占用（约8-10%）、仓储运营、保险及商品损耗风险。

缺货成本系数 (3倍毛利)：这是为应对高波动、脉冲式需求特性的关键假设。对于此类商品，缺货不仅损失单笔交易利润，更可能因促销机会错失或顾客体验下降而带来数倍于毛利的长期隐性损失。该系数旨在量化此放大效应。

动态成本计算：模型并非使用统一的成本绝对值，而是基于每个单品的实际历史平均售价来动态计算其单位持有成本与缺货成本。这确保了成本分析能反映商品间的价值差异。

模拟的再订货点范围：成本模拟不仅测试了由报童模型生成的几个理论点，更围绕日均需求 (μ) 设置了从50%到300%的宽泛测试区间。这一设计的目的是确保在U形成本曲线上，能完整捕捉到最低成本点，避免因测试点不足而错过最优策略。

总而言之，所有假设均服务于一个目标：在承认数据“高波动、间歇性”本质的前提下，构建一个既反映普遍商业规则、又可让管理者根据自身风险偏好和实际成本进行校准的决策分析框架。
### 3.5库存策略模拟与优化
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
基于库存策略模拟结果，我们可以清晰地看到提升客户服务水平所需付出的具体库存代价，以及不同商品与门店间的策略差异。其揭示的核心规律是服务水平与库存成本之间存在“加速权衡”关系，即希望进一步降低缺货风险时，所需增加的安全库存会呈现递增趋势。例如，对于CA_1店的FOODS_3_120商品，将服务水平从95%提升至98%所需增加的安全库存，远高于从80%提升至85%的增量，这表明在高端服务水平区间，边际成本急剧上升，管理者必须审慎评估追求极致服务水平的经济性。

进一步对比不同商品发现，需求波动性（CV值）更高的商品，其库存管理代价也相应更高。例如，波动性更大的FOODS_3_090要达到95%的服务水平，其安全库存需达到日均销量的2.0倍，高于FOODS_3_120的1.9倍。这凸显出对于高波动性“难题”商品，需要投入更多的管理关注与资金成本。同时，门店间的规模差异直接决定了库存水位的绝对值，例如同一商品在核心门店CA_3的再订货点（515件）远高于在CA_1的门店（268件），这强有力地说明绝不能在不同门店间套用统一的库存标准。

综上所述，本次模拟的核心价值在于将“服务水平”这一抽象的管理目标，转化为“安全库存”与“再订货点”这两个可具体执行的仓储参数，并量化了其间的权衡关系。它提供的并非一个确定的答案，而是一份清晰的决策菜单，明确了达成不同服务水平目标所需的库存资源投入，为管理层结合成本权衡分析进行最终判断，并实施差异化的精细管控提供了关键的数据基础。
### 3.6成本权衡分析
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
成本模拟结果构成了本次分析中最具决策价值的产出，它如同一份详尽的“经济体检报告”，清晰地揭示了不同库存策略下的真实财务后果。解读这一结果，我们首先从微观层面审视单条策略记录所呈现的完整“财务账单”：以CA_1店的FOODS_3_120为例，一个较低的再订货点（ROP=119）策略虽然将库存持有成本压至极低，却引发了占总成本99%的巨额缺货损失，最终在模拟期内付出了超过31万元的总成本，而实际服务水平仅达94%。这深刻揭示了片面追求低库存持有成本可能导致“省小钱、亏大钱”的战略失误。

进一步地，跟踪单个商品随再订货点变化的成本趋势，揭示了一条经典的U形成本曲线与边际效益递减规律。该曲线可划分为三个特征鲜明的阶段：在“缺货主导区”，增加库存能显著降低缺货成本，服务水平和成本效益快速提升，属于“花小钱办大事”；进入“平衡最优区”后，缺货成本趋零，总成本在持有成本的平缓增加中达到并稳定于最低点，例如将CA_1店FOODS_3_120的ROP从119件提升至156件，即可实现从时有缺货到永不缺货的质变，同时日均成本下降44%，这是关键的帕累托改进；一旦步入“过度库存区”，持有成本成为主导并驱动总成本回升，意味着库存投入的边际收益已转为负值。

基于此规律进行跨商品与跨门店对比，我们得出了差异化的最优策略图谱。对于高波动性的流量型商品FOODS_3_090，需采用“以量稳价”策略，在CA_1和CA_3店分别维持约268-301件和515-576件的最优库存，其庞大的销量规模摊薄了单位成本，显示出优异的成本效益。对于利润型商品FOODS_3_120，则需“精准高备”，在CA_1店的最优库存区间为156-175件。值得注意的是，核心门店CA_3因需求规模更大，其各商品的最优库存绝对值显著高于CA_1，这强有力地驳斥了在不同门店套用统一库存标准的做法。

综上所述，本模拟的核心价值在于将“提升服务水平”的抽象目标，转化为“再订货点”和“日均总成本”这两个可精准执行与考核的运营参数。最终的管理启示是，库存策略的目标应从追求固定的服务水平百分比，转向寻求并锁定使总成本最低的再订货点。据此，我们可下达明确指令：例如，将CA_1店FOODS_3_120的库存警报线设置为156件，目标是将与该商品相关的日均库存总成本控制在30元以内。这标志着库存管理从经验判断迈向数据驱动的科学决策。
###  3.7成本-服务水平权衡
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
成本-服务水平权衡分析揭示了一个深刻且反直觉的管理洞见：对于所分析的核心单品，单纯追求更高的理论服务水平，在经济效益上可能既不必要，也非最优。核心发现在于，理论服务水平与实际达成的服务水平之间存在显著差异。例如，为CA_1店的FOODS_3_120设计理论值为80%的策略，在历史模拟中实际实现了100%的现货率，这表明基于历史波动性的经典公式计算可能偏于保守。更关键的规律是边际效益递减现象：当理论服务水平提升至高位（如98%）后，继续追加库存投资对提升实际客户体验已无贡献，反而会推高成本。例如，将该商品的理论服务水平从98%提升至99%，需增加13件再订货点，日均成本从18.17元上升至18.79元，而实际服务水平早已稳定在100%。这一“性价比拐点”的存在，意味着存在一个成本效益最优的区间。

进一步分析显示，各单品实现最低日均成本所对应的理论服务水平均在98%或99%，说明为这些高波动性商品维持极高的现货率本身具有经济合理性，但追求极致（如99.9%）则可能成本过高。具体而言，利润波动型的FOODS_3_120在98%服务水平达到成本最低点，而流量明星型的FOODS_3_090则在99%达到成本最低点，且因其巨大的销售规模，其绝对成本更低，承担更高保障级别的经济性更好。

因此，最终的管理决策不应再围绕“设定多高的服务水平目标”这一抽象问题展开，而应转化为一个具体的权衡：“我们愿意为最后1%的理论安全边际，支付多少额外的成本？”决策者应基于每个商品独特的“成本-服务”曲线，精准定位其“性价比拐点”，其核心原则是在确保实际客户体验（接近100%现货率）卓越的前提下，追求库存相关总成本的最小化，而非刻板地追求理论服务水平的数值。
### 3.8管理建议
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
这份最终的管理建议表是整个供应链分析的核心交付成果，旨在直接回应“具体如何执行”的管理诉求。解读此表需结合之前的成本模拟数据，以充分理解其背后的权衡逻辑与真实业务含义。表格推荐对所有四个商品采用99%的服务水平并给出了具体的再订货点，其核心逻辑在于算法优先选择了“在能实现100%实际服务水平的所有策略中总成本最低的那一个”，而非全局成本最低点。这意味着该建议将“杜绝缺货”置于“成本绝对最小化”之上，是一种在“零缺货”硬约束下的最优解。例如，对于CA_1店的FOODS_3_120，建议的ROP为239.8件，这虽会导致库存持有成本上升，但旨在确保几乎永不缺货，其对应的日均成本为9.44元。

进一步对比各商品发现，尽管服务水平目标相同，但不同商品的成本效益差异巨大。销量巨大、规模效应显著的FOODS_3_090（特别是CA_3店）维持高库存的“单位成本”极低，是值得投资的“优等生”；而销量相对较低、波动性高的FOODS_3_120，其维持高服务水平的相对成本则高得多，属于“成本敏感户”。这揭示了实施差异化库存策略的内在必要性。

因此，面对此建议，理性的管理行动并非直接采纳，而应启动一个校准与决策的闭环：首先，必须审阅并校准模型依赖的关键财务假设（如持有成本与缺货损失的真实比率），因为最优解会随假设变动；其次，需明确公司战略是优先“成本领先”还是“服务卓越”，这决定了是选择全局成本最优的激进策略，还是采纳当前保守的零缺货方案；继而，应实施差异化策略，对成本效益好的商品可采纳建议，对成本敏感的商品则可考虑更激进的方案；最后，必须建立监控与反馈机制，将建议值作为初始参数，在实际运营中紧密跟踪其服务水平和成本表现，并持续动态调整。

总之，这份管理建议表并非一个不容置疑的标准答案，而是一份基于特定管理偏好（零缺货优先）的数据驱动强参考方案。它的核心价值在于量化了“零缺货”的具体代价，揭示了不同商品策略的天然差异，从而将管理决策从经验直觉层面提升到了基于数据与明确权衡的科学讨论层面。最终决策应是企业战略、实际财务成本与本数据建议三者综合的产物。
### 3.9结论和建议
**结论**
本次单品微观诊断，通过对加州核心门店CA_1与CA_3中两个核心单品（FOODS_3_090与FOODS_3_120）的深度剖析，完成了从宏观市场定位到可执行库存参数的最终转化。诊断揭示，这两个贡献主要销售额的“核心畅销品”呈现出高频、高波动、间歇性与脉冲式的复杂需求本质，其变异系数(CV)高达0.77以上，并伴有高达20%的零销日和长达数月的连续无销售期。需求分布可视化进一步证实，其历史数据严重偏离经典库存模型所依赖的正态分布假设，展现出显著的“零值堆积”与“右偏长尾”特征。这一根本性发现意味着，直接套用基于稳定、连续需求假设的传统模型将严重脱离实际，无法有效管理此类商品。

然而，通过构建“基线+事件”的双轨分析框架，并在模拟中引入真实的财务成本（持有成本与缺货损失），我们成功量化了不同库存策略的经济后果。分析清晰地绘制出各单品的U形成本曲线，并精准定位了其“性价比拐点”。核心规律表明，存在一个最优的再订货点（ROP）区间，能在实现接近100%实际服务水平的同时，使库存相关的日均总成本最小化。例如，对于CA_1店的FOODS_3_120，将ROP设置在156-175件，即可达成成本与服务的卓越平衡。这一过程强有力地证明，尽管需求模式复杂，但通过数据驱动的模拟优化，依然能够为管理层提供清晰、量化且差异化的决策依据，从而将供应链管理从应对不确定性的艺术，提升为基于历史数据进行科学规划与主动响应的学科。

**建议**

基于以上结论，我们提出涵盖策略框架、具体行动与持续优化三个层面的综合性建议。

首先，在策略框架上，我们正式建议采纳 “基线+事件”双轨库存管理系统。该系统要求将库存决策分离为两个层面：一是为应对日常的“基线需求”，使用优化后的经典模型（基于本次模拟得出的ROP）设定常态库存水位；二是必须与营销日历深度集成，建立“事件驱动”的库存预案机制，在已知的促销或旺季来临前，主动、动态地切换至更高的“事件安全库存”水平。这一框架是对抗脉冲式需求、实现主动供应链管理的核心。

其次，在具体行动上，应立即实施差异化的库存参数设定。我们建议以本次成本模拟找到的最优ROP区间作为初始执行标准，例如，将CA_1店FOODS_3_120的库存警报线设置为156件，并将与该商品相关的日均库存总成本管控目标定为30元以内。同时，必须彻底放弃在不同门店或不同商品间套用统一“覆盖X天”库存规则的粗放做法，转而依据每个商品-门店组合独特的成本曲线与业务画像（如“流量型”或“利润型”）进行精准配置。对于FOODS_3_090等高销量、高效益商品，可采纳更保守的库存策略以保障核心流量；对于FOODS_3_120等成本敏感商品，则应严格围绕成本最优区间操作。

最后，建立管理校准与持续优化的闭环机制至为关键。本次分析的所有建议均基于一组标准的成本假设（如20%年持有成本率、3倍缺货损失）。在实施前，管理层必须结合企业真实的财务数据（资金成本、仓储费率、缺货对品牌和销售的长期影响）对这些假设进行评审与校准。建议将本次推荐的ROP作为试点运行的基准，并配套建立关键绩效指标（KPI）监控体系，紧密追踪实际达成的服务水平和发生的真实成本。在1-2个完整的补货或销售周期后，应根据实际运营数据对模型进行反馈与调优，从而形成一个从“数据诊断”到“策略执行”再到“效果反馈”的持续改进循环，最终推动供应链库存管理迈向真正的智能化与精细化。
## 4.总结
本次供应链需求与库存分析，通过一个严谨的“从面到点”三层漏斗框架，成功地将沃尔玛长达五年的海量销售历史数据，转化为了一系列清晰、可执行的管理洞察与决策指令。整个分析历程始于宏观的战略俯瞰，历经中观的网络诊断，最终落子于微观的精准优化，完成了从“认知市场”到“指导仓库”的完整闭环。

在宏观层面，我们确立了FOODS品类作为公司基本盘的战略地位，其需求整体稳定且呈现显著的季节性规律。更重要的是，我们精准识别出加利福尼亚州（CA）作为贡献全国超四成销售的核心市场，这为所有后续的资源倾斜与深度优化提供了无可争议的战略依据。

聚焦于核心区域网络，分析揭示了加州供应链系统的复杂性与管理挑战的根源。量化研究虽然验证了集中库存能平滑需求波动的经典理论，但更关键的发现是：各门店需求存在高度正相关性，极大地削弱了“风险共担”的潜在效益；同时，网络内部存在着“现金牛”、“问题明星”、“风险门店”与“长尾门店”等需求模式迥异的节点。这两大发现共同指向一个核心结论：传统的、一刀切的供应链策略在此网络中将必然失效，差异化与精准化是提升网络整体效率与韧性的唯一路径。

在微观执行层面，通过对代表性门店与核心单品的深度诊断，我们直面了现实业务中最棘手的难题：那些贡献主要销售额的“畅销品”，其需求本质是高频、高波动、间歇性的脉冲模式，严重偏离了经典库存模型的假设。然而，通过构建“基线+事件”的双轨分析框架，并引入财务成本进行模拟优化，我们成功驾驭了这种复杂性。分析不仅量化了服务水平与库存成本之间“加速权衡”的规律，更精准定位了使总成本最小化的经济最优再订货点（ROP），例如为CA_1店的FOODS_3_120商品明确了156-175件的最优库存区间。

综上所述，本项研究的终极价值不仅在于产出针对具体商品（FOODS_3_090、FOODS_3_120）在具体门店（CA_1、CA_3）的、数据驱动的库存参数建议（如再订货点、安全库存及目标成本），更在于完整示范了一套可复用的数据分析方法论。它深刻阐明，在复杂多变的零售环境中，卓越的供应链管理绝非依赖直觉或单一公式，而必须建立在对多层次数据进行系统性解构与重构的基础之上——从宏观趋势中锁定战场，从网络结构中识别规则例外，最终在微观脉冲中寻得成本与服务的最优平衡。本次分析所形成的策略框架与决策流程，为将数据资产持续转化为竞争优势，奠定了坚实的基础。
## 5.可视化看板
<img width="1169" height="649" alt="861d1ed5ba68563798dc650d698f4ae4" src="https://github.com/user-attachments/assets/24ab2000-379c-447d-85f7-0ef96d4cc523" />
<img width="1167" height="658" alt="2cd9033c2a4efb69afca903673f62e42" src="https://github.com/user-attachments/assets/d3e82921-a1a9-4585-a9e1-be79e0f913a0" />
<img width="1165" height="654" alt="513924fe2e8ce1d1dfcc675745b8c17c" src="https://github.com/user-attachments/assets/d04d4a64-281d-4052-bef0-ae2ca15be361" />










