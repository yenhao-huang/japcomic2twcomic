# 文字分割策略說明

`TextAllocater` 提供了四種文字分割策略,用於將翻譯文字智能分配到多個 bounding boxes。

## 使用方法

```python
from lib.text_allocater.text_allocater import TextAllocater

allocater = TextAllocater(
    translation_json_path="results/translation/hunyuan.json",
    text_region_json_path="results/text_detection/text_region.json",
    output_path="results/text_allocate/text_allocate.json",
    split_strategy="region_newline"  # 選擇策略
)

results = allocater.run()
```

## 策略詳解

### 1. 均分策略 (even)
**適用場景**: 文字長度均勻,沒有明顯的語意分界

**運作方式**:
- 根據字數將文字平均分配到每個 box
- 簡單快速,但可能在單詞或語句中間切斷

**範例**:
```
輸入: "這是一個測試文字"
Boxes: 3 個
輸出: ["這是一", "個測試", "文字"]
```

**優點**:
- 簡單可靠
- 每個 box 的文字長度接近

**缺點**:
- 可能破壞語意完整性
- 在標點符號或單詞中間切斷

---

### 2. 換行策略 (newline)
**適用場景**: 翻譯文字包含換行符號,每行代表一個語意單元

**運作方式**:
- 優先根據換行符號 `\n` 分割文字
- 如果行數 = box 數量:直接一一對應
- 如果行數 < box 數量:分割最長的行
- 如果行數 > box 數量:合併相鄰的行

**範例**:
```
輸入: "第一行\n第二行\n第三行"
Boxes: 3 個
輸出: ["第一行", "第二行", "第三行"]
```

**優點**:
- 保持換行結構
- 適合詩歌、對話等格式化文字

**缺點**:
- 如果換行不規則,可能分配不均
- 依賴翻譯模型的換行輸出

---

### 3. Region + 換行策略 (region_newline) ⭐ 推薦
**適用場景**: 漫畫翻譯 - boxes 有空間分群,文字有換行結構

**運作方式**:
1. **空間分析**: 分析 boxes 的位置,將距離較遠的 boxes 分成不同 regions
2. **距離閾值**: 使用 1.5 倍平均距離作為 region 分界
3. **換行分配**: 為每個 region 分配對應數量的文字行
4. **智能調整**: 如果分配不均,使用均分策略補充

**範例**:
```
Boxes 位置:
  Box1: (100, 50)  ←─ Region 1
  Box2: (100, 80)  ←┘

  Box3: (100, 200) ←─ Region 2 (距離較遠)
  Box4: (100, 230) ←┘

輸入: "對話1\n對話2\n對話3\n對話4"
輸出:
  Box1: "對話1"  (Region 1)
  Box2: "對話2"
  Box3: "對話3"  (Region 2)
  Box4: "對話4"
```

**優點**:
- 考慮空間佈局和文字結構
- 適合漫畫對話框的自然分布
- 保持語意和視覺的對應關係

**缺點**:
- 需要合理的 box 位置分布
- 對換行符號有一定依賴

**參數說明**:
- 垂直排列 (y_variance > x_variance): 根據 y 座標分群
- 水平排列: 根據 x 座標分群
- Region 分界閾值: 1.5 × 平均相鄰距離

---

### 4. 智能分割策略 (smart)
**適用場景**: 長文字需要保持語意完整性

**運作方式**:
1. **標點符號優先級**:
   - 高: 句號、驚嘆號、問號 (`。!?`)
   - 中: 逗號、分號 (`、,;`)
   - 低: 換行符號、空白

2. **分割邏輯**:
   - 尋找接近理想分割位置的標點符號
   - 優先在句子邊界分割
   - 保持每個 box 的語意完整

3. **Fallback**: 如果找不到合適的標點符號,使用均分策略

**範例**:
```
輸入: "這是第一句。這是第二句。這是第三句。"
Boxes: 3 個
輸出: [
  "這是第一句。",
  "這是第二句。",
  "這是第三句。"
]
```

**優點**:
- 保持語意完整
- 每個 box 包含完整的句子或語句
- 適合長文字、旁白、說明文字

**缺點**:
- 如果標點符號分布不均,可能導致長度差異大
- 對中文/日文標點符號的依賴

---

## 策略選擇建議

| 場景 | 推薦策略 | 理由 |
|------|---------|------|
| 漫畫對話框 | `region_newline` | 考慮空間佈局和對話結構 |
| 多行詩歌 | `newline` | 保持原有換行格式 |
| 長篇旁白 | `smart` | 保持語意完整性 |
| 簡短文字 | `even` | 簡單快速 |
| 不確定 | `region_newline` | 最全面的策略 |

## 實現細節

### 關鍵方法

1. **[_split_text_even](lib/text_allocater/text_allocater.py:349)**: 均分策略
2. **[_split_text_by_newline](lib/text_allocater/text_allocater.py:381)**: 換行策略
3. **[_split_text_by_region_newline](lib/text_allocater/text_allocater.py:435)**: Region + 換行策略
4. **[_split_text_smart](lib/text_allocater/text_allocater.py:543)**: 智能分割策略
5. **[_cluster_boxes_into_regions](lib/text_allocater/text_allocater.py:505)**: Box 空間分群

### 向後兼容

原有的 `_split_text_by_boxes()` 方法仍然保留,現在會調用 `_split_text_even()`,確保舊代碼正常運作。

## 測試範例

```python
# 測試不同策略
strategies = ["even", "newline", "region_newline", "smart"]

for strategy in strategies:
    allocater = TextAllocater(
        translation_json_path="results/translation/hunyuan.json",
        text_region_json_path="results/text_detection/text_region.json",
        output_path=f"results/text_allocate/text_allocate_{strategy}.json",
        split_strategy=strategy
    )
    results = allocater.run()
    print(f"{strategy}: {len(results)} images processed")
```
