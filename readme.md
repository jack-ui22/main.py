# 信息检索实验报告

## 实验一：信息检索

### 小组信息


### 云盘链接

---

## 一、实验概述

### 1.1 实验目标
1. 掌握信息检索系统的基本架构和工作原理
2. 学习并实现倒排索引结构及其优化方法
3. 熟悉跳表指针技术在信息检索中的应用及性能优化
4. 学习文本预处理、分词、停用词过滤等信息检索基础技术
5. 掌握倒排索引的压缩存储和分块管理技术
6. 分析不同搜索优化方法的性能差异并进行优化

### 1.2 实验环境
- 编程语言：Python 3.12
- 分词库：nltk
- 停用词表：哈工大停用词表（对内容稍加更改）

### 1.2 数据集说明
- **数据来源**：课程组提供的Meetup数据集
- **重点处理文件**：Event类XML文件
- **数据规模**：总文档数：93152  清理后大小：195MB

---

## 二、文档解析与规范化处理

### 2.1 文档解析方法
文档解析通过Python的re处理库和正则表达式匹配，提取关键信息。系统从XML文件中读取预处理后的事件数据，包括关键词和描述信息。并清理相关的xml语句

提取关键词结构如下：
```python
patterns = {
                "key_word": file_name,
                "event_id": r"</city><id>(?P<event_id>.*?)</id><country>",
                "event_name": r"<name>(?P<name>.*?)</name>",
                "event_time": r"<time>(?P<event_time>.*?)</time>",
                "event_url": r"<event_url>(?P<event_url>.*?)</event_url>",
                "description": r"<description>(?P<description>.*?)</description>",
                "member_id": r"<member_id>(?P<member_id>.*?)</member_id>",
                "join_mode": r"<join_mode>(?P<join_mode>.*?)</join_mode>",
                "city": r"<city>(?P<city>.*?)</city>",
                "country": r"<country>(?P<country>.*?)</country>"
            }
```
提取过程：    
```python
event_data = {}
            for key, pattern in patterns.items():
                if key == 'key_word':
                    event_data[key] = pattern
                    continue

                match = re.search(pattern, xml_content, re.DOTALL)
                if match:

                    event_data.update(match.groupdict())
                else:
                    group_names = re.findall(r'\(\?P<(\w+)>', pattern)
                    for group_name in group_names:
                        event_data[group_name] = None
            if event_data.get('description') is not None:
                event_data['description'] = self.clear_xml(text=event_data['description'])
            if event_data.get('member_id') is not None:
                match = re.search(r"PastEvent (?P<name>.*?)\.xml", file_name)
                if match:
                    new_id = match.group('name')

                    event_data['rspvs_member'] = self.members_info(id=new_id)
                else:print("提取失败!")
            else:
                event_data['rspvs_member'] = None

            all_events_data.append(event_data)

        df_events = pd.DataFrame(all_events_data)
        df_events.to_json('events_data.json', orient='records', indent=4, force_ascii=False)
        return df_events 
```

- 通过对典型的xml语法及其其余典型多余部分的正则匹配，有效清除大部分无用的消息
清理过程如下：

```python
    def clear_xml(self, text):
        if not text:
            return ""
        text = re.sub(r'<[^>]+>', '', text)
        replacements = {
            '&amp;': '&',
            '&lt;': '<',
            '&gt;': '>',
            '&quot;': '"',
            '&apos;': "'",
            '&#39;': "'",
            '&#34;': '"',
            '&#38;': '&',
            '&#60;': '<',
            '&#62;': '>',
            '&nbsp;': ' ',
            '&copy;': '(c)',
            '&reg;': '(R)',
            '&trade;': '(TM)',
        }
        for entity, replacement in replacements.items():
            text = text.replace(entity, replacement)
        # 3. 移除 CDATA 部分
        text = re.sub(r'<!\[CDATA\[(.*?)\]\]>', r'\1', text, flags=re.DOTALL)
        text = re.sub(r'<\?.*?\?>', '', text, flags=re.DOTALL)
        # 6. 清理多余空格和换行
        text = re.sub(r'\s+', ' ', text)  
        text = text.strip()
        return text
```

- 将提取的文本通过保存为json文本  生成待检索文档events_data.json 

### 2.2 文本预处理流程

#### 2.2.1 分词处理

采用NLTK库进行分词处理，将文本分解为单词序列，为后续的词项提取做准备。

#### 2.2.2 规范化处理
- **大小写归一化**：将所有文本转换为小写形式，消除大小写差异
- **进一步清理文档**：通过跟精准的正则匹配，消除文本的提取token过程中的xml，网址等无用的token。
- **停用词去除**：使用stopwords.txt文件定义的停用词列表，过滤掉无意义的常用词
- **特殊字符处理**：移除标点符号、数字和其他非字母字符
- **词干提取/词形还原**：未实现词干提取，但通过基本规范化处理确保词项一致性

### 2.3 预处理效果示例
**原始文本** ：`A recipe exchange!<br \/>Ein Rezept Austausch-<br \/>Everyone eats, and they would like to enjoy one of your favorite recipes!`
**处理后结果** ：`['recipe', 'exchange', 'rezept', 'austausch', 'eats', 'enjoy', 'favorite', 'recipes']`

---

## 三、倒排索引的构建与优化

### 3.1 倒排表构建
#### 3.1.1 数据结构设计
采用字典结构存储倒排索引，格式为：`inverted = {'token': None,'file_name': None}`

#### 3.1.2 构建算法
1. 读取预处理后的JSON数据
2. 对每个文档进行分词和清理
3. 遍历词项，记录其在文档的关键词
4. 更新倒排索引字典
5. 保存索引结构以便后续使用

### 3.2 跳表指针设计
#### 3.2.1 跳表指针实现
实现了单层和双层跳表指针结构，用于加速倒排索引的查询过程。

```python
# 单层跳表指针实现
    def create_one_skip_pointer(self):
        if not hasattr(self, 'tokens'):
            print('倒排表未加载')
            return
        skip_list = {}
        block_size = max(1, int(math.sqrt(self.line_number)))
        i = 0
        while i < self.line_number:
            start = i
            i += block_size
            end = min(i, self.line_number) - 1
            skip_list[start] = {
                'type': 'level1',
                'end': end,
            }
        return skip_list
```
```python
#双层跳表实现
    def create_two_skip_pointer(self):
        '''二级指针'''
        if not hasattr(self, 'tokens'):
            print('倒排表未加载')
            return
        start_time = time.time()
        line_number = self.line_number
        level1_block_size = max(1, int(math.sqrt(line_number)))
        level1_count = math.ceil(line_number / level1_block_size)
        level2_gap = max(1, int(math.sqrt(level1_count)))
        skip_list = {}
        current = 0
        while current < line_number:
            end = min(current + level1_block_size - 1, line_number - 1)
            skip_list[current] = {
                'type': 'level1',
                'end': end,
                'level2': []
            }
            # 跳转到下一个块
            current = end + 1
        sorted_positions = sorted(skip_list.keys())
        for i in range(len(sorted_positions)):
            current_pos = sorted_positions[i]
            current_block = skip_list[current_pos]

            for j in range(i + level2_gap, min(i + 5 * level2_gap, len(sorted_positions)), level2_gap):
                jump_pos = sorted_positions[j]
                current_block['level2'].append(jump_pos)
        
        end_time = time.time()
        self.two_skip_create_time = end_time - start_time
        return skip_list
```

#### 3.2.2 单层和双层跳表指针的性能和开支差异
| 指针类型 | 步长配置 | 存储空间(字节) | 查询性能（s） |
|---------|---------|---------|---------|
| 单层指针 | $$len^{0.5}$$  | 136016  | 0.015168|
| 单层指针 | $$len/100$$ | 42664 | 0.023073 | 
| 单层指针 | $$len/1000$$ |416336  | 0.026256| 
| 双层指针 | $$(len/10,num^{0.5})$$| 5366| 0.006568 | 
| 双层指针 | $$(len/10,num/100)$$ | 5430 | 0.007380  |
| 双层指针 | $$(len/10,num/1000) $$| 5142  |0.004837 |

**说明：** len=112980，num为双层指针第一层数量,(m,n)表示第一层步长m,第二层步长为n  查询性能是随机300个token查询所用的时间（不包括计算跳表指针的时间），产生倒排表时间约等于0 ，故没有记录。

**性能分析总结：**
- **存储效率**：双层指针相比单层指针节省很大的存储空间
- **查询性能**： 双层指针比较单层指针有较快的查询性能
- **步长影响**：增大步长可进一步压缩存储，但查询性能略有下降

表格已按不同步长配置详细对比单双层跳表的性能差异，准确记录了存储、构建时间和查询效率的权衡关系。


### 3.3 索引优化措施
#### 3.3.1 倒排表优化

- 位置信息以整数列表形式存储，每个元素代表词项在文档中出现的位置偏移量，用于支持短语检索功能。倒排表数据结构改进为：`{token: {file_name: [positions]}}`

实现代码：

``` python
    def inverted_list(self):
        """
        倒排表格式
        {token: {file_name: [positions]}}
        """
        try:
            inverted_index = {}
            df = pd.read_json(self.json_name)
            stopwords = self.read_stopwords()
            key_words = df['key_word'].tolist()
            decs_main = df['description'].tolist()
            for i in tqdm(range(len(key_words))):
                key_word = key_words[i]
                dec_main = decs_main[i]
                if dec_main is None:
                    continue
                finall_key_word = re.findall(
                    r"PastEvent (?P<name>.*?).xml", key_word
                )
                file_name = finall_key_word[0] if finall_key_word else "Unknown"
                tokens = nltk.word_tokenize(dec_main)
                filtered_tokens = self.clean_token(tokens=tokens, stopwords=stopwords)
                # 创建token位置映射
                token_positions = {}
                for pos, token in enumerate(filtered_tokens):
                    if token not in token_positions:
                        token_positions[token] = []
                    token_positions[token].append(pos)
                for token, positions in token_positions.items():
                    if token not in inverted_index:
                        inverted_index[token] = {}

                    inverted_index[token][file_name] = positions

            return inverted_index

        except FileNotFoundError:
            print('json not found')
            return {}
        except Exception as e:
            print(f"error in read json:{e}")
            return {}

```
- 对词典和倒排表分开存储，以便后续压缩

实现代码：
``` python
def save_separate_structures(self, inverted_index):
        try:
            dict_file = "词典_前端编码.txt"  # 压缩后的词典
            original_dict_file = "词典_原始.txt"  # 原始词典
            postings_file = "倒排表_差值编码.csv"  # 压缩后的倒排表
            original_postings_file = "倒排表_原始.csv"  # 原始倒排表
            sorted_tokens = sorted(inverted_index.keys())
            with open(original_dict_file, 'w', encoding='utf-8') as f:
                for token in sorted_tokens:
                    f.write(f"{token}\n")
            encoded_dict = self.front_coding(sorted_tokens)
            with open(dict_file, 'w', encoding='utf-8') as f:
                for common_len, data in encoded_dict:
                    f.write(f"{common_len}:{data}\n")

            with open(original_postings_file, 'w', encoding='utf-8') as f:
                f.write("token,file_name,positions\n")
                for token, file_positions in inverted_index.items():
                    for file_name, positions in file_positions.items():
                        positions_str = ','.join(map(str, positions))
                        f.write(f"{token},{file_name},{positions_str}\n")
            with open(postings_file, 'w', encoding='utf-8') as f:
                f.write("token_id,file_name,encoded_positions\n")
                for token_id, token in enumerate(sorted_tokens):
                    for file_name, positions in inverted_index[token].items():
                        encoded_positions = self.delta_encoding(positions)
                        positions_str = ','.join(map(str, encoded_positions))
                        f.write(f"{token_id},{file_name},{positions_str}\n")
```

#### 3.3.2 索引压缩方法
**差值编码**
- **实现细节**：将对词项位置信息 除第一个外其余的与前一个的差值作为位置保存，有效降低保存数据的大小

实现代码:
```python
ef delta_encoding(self, positions):
        """
        对位置列表进行差值编码
        """
        if not positions:
            return []
        
        encoded = [positions[0]]
        for i in range(1, len(positions)):
            encoded.append(positions[i] - positions[i-1])
        return encoded
```
```python    
    def delta_decoding(self, encoded_positions):
        """
        对差值编码的位置列表进行解码
        """
        if not encoded_positions:
            return []
        
        decoded = [encoded_positions[0]]
        for i in range(1, len(encoded_positions)):
            decoded.append(decoded[i-1] + encoded_positions[i])
        return decoded
```
- **压缩效果**: 倒排表有原来的92700.33 KB压缩到84707.69 KB ,压缩8.62%。

**方法二：前端编码**

- **实现细节**：将词典按字典序排序后，每n个词项分为一组，提取组内词项的最长公共前缀并只存储一次。每组存储结构为：公共前缀长度（1字节）、公共前缀字符串、以及组内每个词项的后缀部分（每个后缀以长度前缀或空字符结尾）。查询时，通过二分查找在组内进行匹配。

``` python
    def front_coding(self, tokens):
        """
        对排序后的词项列表进行前端编码压缩
        """
        if not tokens:
            return []
        
        encoded_dict = []
        i = 0
        block_size = 8
        while i < len(tokens):
            prefix = tokens[i]
            encoded_dict.append((-1, prefix))
            j = 1
            while i + j < len(tokens) and j < block_size:
                current_token = tokens[i + j]
                common_len = 0
                min_len = min(len(prefix), len(current_token))
                while common_len < min_len and prefix[common_len] == current_token[common_len]:
                    common_len += 1
                suffix = current_token[common_len:]
                encoded_dict.append((common_len, suffix))
                j += 1
            i += j
        return encoded_dict
```
```python  
    def front_decoding(self, encoded_dict):
        """
        对前端编码的词典进行解码
        """
        tokens = []
        prefix = None
        
        for item in encoded_dict:
            common_len, data = item
            if common_len == -1:
                prefix = data
                tokens.append(prefix)
            else:
                if prefix is not None:
                    full_token = prefix[:common_len] + data
                    tokens.append(full_token)
        
        return tokens
```

- **压缩效果**：当n=8时，有最优压缩效果，词典大小从1523.99 kb 压缩到1332.99 KB  压缩12.53%。

示例：
```
压缩前：
['app', 'appair', 'appal', 'appalachia', 'appalachian', 'appalachians', 'appall','appointment'])

压缩后：
[(-1, 'app'), (3, 'air'), (3, 'al'), (3, 'alachia'), (3, 'alachian'), (3, 'alachians'), (3, 'all'), (3, 'ointment')]
```

### 3.4 性能对比分析
| 索引类型 | 存储空间 | 构建时间 | 备注 |
|---------|---------|---------|------|
| 原始索引 | 92MB | 8.393s| 原始倒排表 |
| 压缩后索引 | 84MB | 9.121s | 按块存储+前端编码 |

**总结：**  压缩前后，存储空间明显减少，构建倒排表时间稍微增大。

---

## 四、信息检索实践

### 4.1 布尔检索
#### 4.1.1 布尔检索实现

- **原理：** 通过词法分析将查询字符串分解为词项和运算符；使用运算符优先级并生成后缀表达式；从倒排索引中获取每个词项对应的文档列表；根据后缀表达式顺序执行相应的集合操作（交集、并集、补集）；最终返回满足所有逻辑条件的文档集合。系统还支持在压缩和未压缩两种倒排索引结构上进行查询，并提供了详细的查询解析过程和性能统计信息。
- **说明：** 为方便输入，本次实验用'+'代表并集，'*'交集，'-'补集
- **实现：**
    - 并集：
        ```python
              def or_search(self, result1, result2):
        """OR操作：求两个结果集的并集"""
        result = []
        i = j = 0

        while i < len(result1) and j < len(result2):
            file1 = result1[i]['file_name']
            file2 = result2[j]['file_name']

            if file1 == file2:
                result.append(file1)
                i += 1
                j += 1
            elif file1 < file2:
                result.append(file1)
                i += 1
            else:
                result.append(file2)
                j += 1
        while i < len(result1):
            result.append(result1[i]['file_name'])
            i += 1

        while j < len(result2):
            result.append(result2[j]['file_name'])
            j += 1

        return result
        ```
    - 交集：
    ```python
        def and_search(self, result1, result2):
        """AND操作：求两个结果集的交集"""
        result = []
        i = j = 0

        while i < len(result1) and j < len(result2):
            file1 = result1[i]['file_name']
            file2 = result2[j]['file_name']

            if file1 == file2:
                result.append(file1)
                i += 1
                j += 1
            elif file1 < file2:
                i += 1
            else:
                j += 1

        return result
    ```
    - 补集：
    ```python
        def not_search(self, result1, result2):
        """NOT操作：在result1中但不在result2中的文档"""
        result = []
        i = j = 0

        while i < len(result1):
            file1 = result1[i]['file_name']

            while j < len(result2) and result2[j]['file_name'] < file1:
                j += 1

            if j < len(result2) and result2[j]['file_name'] == file1:
                i += 1
                j += 1
            else:
                result.append(file1)
                i += 1
        return result
    ```
**优先级设置：**`{'+': 1, '-': 1, '*': 2, '(': 0, ')': 0}`


#### 4.1.2 查询设计
- 示例1：`(music live+banana)*apples`
结果如下：
```
complex_search('(music live+banana)*apples',is_compress_invarted=True)
result:
music live: ['81323692', '123486042', ... '25663871', '9314864', '9845716', '23996121', '21841841', '13347957', '56033212', '14484303', '23996581', '16280171', '11083923']
banana: ['108193102', ... '83876212', '8528858', '85924302', '9443610', '9767591', '98078042', '99073222', '9943031', '99842182', 'qrrgnynlblc']
apples: ['10036675', '10097209', '101276652', '103120982', '10371255', '104388052', ... '108193102', ...'8849933', '8891221', '8947694', '9001885', ,... 'wtvjpypdbnb']
复杂查询 '(music live+banana)*apples' 的解析结果: ['music live', 'banana', '+', 'apples', '*']
复杂查询 '(music live+banana)*apples' 的结果: ['108193102']
```
- 示例2：`'(music festival+concert)*outdoor -rain'`
结果如下：
```
music festival: ['22871001', '9380524', '14535377', '17189261', '12100078', '8332689', ...]
concert: ['10018515', '100189522', '100282272', '100282372', '100286102', '100286272', ...]
outdoor: ['10029424', '10049712', '100600412', '10063057', '10063091', '10063100', '10063106'...']
rain: ['10005412', '10011314', '10011408', '10014582', '10023287', '10048386',...]
复杂查询 'music festival+concert)*outdoor -rain' 的解析结果: ['music festival', 'concert', '+', 'outdoor', '*', 'rain', '-']
复杂查询 'music festival+concert)*outdoor -rain' 的结果: ['10171565', '10485243', '10485269',...]
```
- 示例3：`'(basketball*tournament)+(football*match)-canceled'`
```
basketball: ['100071292', '10049712', '100577962', '10223924', '10315110', '10522224', '105222532', '10576244'...]
tournament: ['101120472', '10138560', '102341182', '104041632', '104646772', '104647622'...]
football: ['101318562', '10251580', '105222532', '106199872', '10860642', '109561672',...]
match: ['10002656', '100174602', '100330612', '10046414', '100714462', '10171565',
canceled: ['100188382', '10029424', '100812132', '101692362', '101791762',...]
解析结果: ['basketball', 'tournament', '*', 'football', 'match', '*', '+', 'canceled', '-']
复杂查询  的结果: ['106500692', '11398936', '115935232', '13517698', '14221861'...]
```
**说明：** 有些词项结果太多，在示例结果中删去大部分。当某些词项在单个词项去检索中返回为空时，则此词项被当作短语去检索。
**总结**
通过优先级定义和后缀表达式的实现，可以较快高效检索复杂的布尔检索任务


#### 4.1.3 短语检索
- **实现方式**：利用位置信息检查词项是否连续出现

- **查询示例**：`"live music"，outdoor sports`
- 查询结果(为方便演示，仅列出前两个结果)
```
示例1：
[{'file_name': '114913142', 'phrase_positions': [(3, 4)]}, 
{'file_name': '25663871', 'phrase_positions': [(22, 23)]},...]
时间：0.024746417999267578
示例2：
[{'file_name': '8932200', 'phrase_positions': [(3, 4)]},
 {'file_name': '12083411', 'phrase_positions': [(479, 480)]}...]
时间：0.020000
```
**总结：** 对于双词项的短语，检索时间可以达到0.2s,表现出较好的查询性能

#### 4.1.4 性能分析
**不同处理顺序的影响**
token的文档长度：`rain: 2735,outdoor: 1445,concert: 914,festival: 1556,music: 6136`
结果：（时间为查询10次平均后的结果）
```
对于and：
文档数量大到小：music*rain*festival*outdoor*concert
time：0.0662 秒
文档数量小到大：music*(rain*(festival*(outdoor*concert)))'
time:0.1229 秒
文档数量大小组合：((music*concert)*festival)*(rain*outdoor)
time:0.0218 秒
对于or：
文档数量大到小：music+rain+festival+outdoor+concert
time： 0.516 秒
文档数量小到大：music*(rain+(festival+(outdoor+concert)))'
time:0.0268  秒
文档数量大小组合：'((music+concert)+festival)+(rain+outdoor)'
time:0.0293  秒
对于not：固定rain和outdoor带“-”号
文档数量大到小：music-rain+festival-outdoor+concert
time：0.0265  秒
文档数量小到大：(festival-rain+(concert-outdoor))'
time:0.0156 秒
文档数量大小组合：'((music+concert)+festival)-(rain+outdoor)'
time:0.0268  秒
```
**总结**
1. **文档数量顺序对性能影响显著**：小到大策略在OR和NOT操作中表现最佳
2. **混合策略优势**：AND操作中大小组合策略表现最优
3. **性能差异明显**：最优与最差策略间存在5-19倍性能差距


## 4.2 向量空间模型实现与分析

### 4.2.2 TF-IDF计算实现

#### 计算公式
- **词频(TF)**：`TF(t,d) = 词项t在文档d中的出现次数`
- **逆文档频率(IDF)**：`IDF(t) = log(总文档数/(包含t的文档数+1)) + 1`
- **TF-IDF值**：`TF-IDF(t,d) = TF(t,d) × IDF(t)`

#### 关键代码实现
```python
def _calculate_idf(self):
    """计算所有词项的IDF值"""
    print('计算idf')
    idf_values = {}
    total_docs = self.total_files
    for token in tqdm(self.inverted_list, desc="计算IDF值", total=len(self.inverted_list)):
        doc_freq = len(self.inverted_list[token])
        idf = math.log(total_docs / (doc_freq + 1)) + 1  
        idf_values[token] = idf
    
    print("IDF值计算完成")
    return idf_values

def _build_document_vectors(self):
    """构建文档的TF-IDF向量"""
    print(f"开始构建文档TF-IDF向量，总词项数: {len(self.inverted_list)}, 总文档数: {len(self.docs_list)}")

    doc_vectors = {}
    for doc in self.docs_list:
        doc_vectors[doc] = {}

    # 反向遍历优化：按词项处理文档
    for token in tqdm(self.inverted_list.items(), desc="处理词项", total=len(self.inverted_list)):
        token_name, token_docs = token[0], token[1]
        idf = self.idf_values.get(token_name, 0)
        for doc in token_docs:
            tf = len(token_docs[doc])
            doc_vectors[doc][token_name] = tf * idf

    # 向量归一化
    for doc in tqdm(doc_vectors.items(), desc="归一化向量", total=len(doc_vectors)):
        doc_name, vector = doc[0], doc[1]
        norm = math.sqrt(sum(val**2 for val in vector.values()))
        if norm > 0:
            for token in vector:
                vector[token] /= norm
    
    print("文档TF-IDF向量构建完成")
    return doc_vectors
```
### 4.2.3 相似度计算实现

#### 余弦相似度算法
**算法原理**：余弦相似度通过计算两个向量夹角的余弦值来衡量其相似度，公式为：
```
cos(A,B) = (A·B) / (||A|| × ||B||)
```
其中A·B表示向量点积，||A||和||B||表示向量的模。

```python
def _cosine_similarity(self, vec1, vec2):
    """计算两个向量的余弦相似度"""
    dot_product = 0
    for token in vec1:
        if token in vec2:
            dot_product += vec1[token] * vec2[token]

    norm1 = math.sqrt(sum(val**2 for val in vec1.values()))
    norm2 = math.sqrt(sum(val**2 for val in vec2.values()))

    if norm1 == 0 or norm2 == 0:
        return 0
    
    return dot_product / (norm1 * norm2)
```

#### 查询向量构建
**实现方法**：基于查询词项构建TF-IDF向量，并进行归一化处理。具体步骤：
1. 统计查询中每个词项的出现频率（TF）
2. 将TF与对应词项的IDF值相乘得到TF-IDF
3. 对结果向量进行归一化处理

```python
def _create_query_vector(self, query_tokens):
    """创建查询向量"""
    query_freq = {}
    for token in query_tokens:
        query_freq[token] = query_freq.get(token, 0) + 1

    query_vector = {}
    for token, freq in query_freq.items():
        tf = freq
        idf = self.idf_values.get(token, 0)
        query_vector[token] = tf * idf
    
    # 查询向量归一化
    norm = math.sqrt(sum(val**2 for val in query_vector.values()))
    if norm > 0:
        for token in query_vector:
            query_vector[token] /= norm
    
    return query_vector
```

### 4.2.4 性能测试结果

#### 查询性能统计
| 查询内容 | 预处理结果 | 相关文档数 | 最高相似度 | 查询时间 |
|---------|-----------|----------|----------|----------|
| live music apple | ['live', 'music', 'apple'] | 8,689 | 0.4968 | 4.7471秒 |
| banana apple | ['banana', 'apple'] | 302 | 0.3975 | 0.0075秒 |
| music festival | ['music', 'festival'] | 7,190 | 0.5900 | 0.1306秒 |
| amazing | ['amazing'] | 1,243 | 0.5202 | 0.0291秒 |

### 4.2.5 总结
查询速度接近布尔查询，查询单个词项有较高的检索效率和检索准确度，但是对于多个词项的查询，如”apple and banana“时 ，如某文档长度较少且apple的出现率较高时，会有较高的相似度


---

## 五、选做部分说明
### 5.1 未实现部分
- 爬虫实践
- 个性化检索
- 基于文档表征的检索

---

## 六、实验总结与展望

### 6.1 实验收获
通过本次信息检索系统实现实验，我们系统地完成了从底层索引构建到高级检索功能的全过程，获得了以下关键收获：

1. **理论与实践结合**：深入理解并实际实现了倒排索引的构建原理、压缩方法和查询优化技术，将抽象的信息检索理论转化为可运行的系统。

2. **核心技术掌握**：熟练运用跳表指针技术（单层和双层）、向量空间模型、TF-IDF权重计算等信息检索核心技术，显著提升了检索效率。

3. **算法性能优化**：通过实践掌握了索引构建、查询处理和结果排序过程中的性能优化策略，能够有效平衡内存占用与查询速度。

4. **多检索模型实现**：成功实现了布尔检索（AND/OR/NOT操作）、短语检索和基于TF-IDF的向量空间检索，理解了不同检索模型的适用场景。


### 6.2 遇到的问题与解决方案

#### 6.2.1 索引构建效率问题
**问题描述**：原始倒排表构建过程中，文档遍历和索引更新操作频繁，导致整体构建速度较慢，特别是处理接近十万级文档时耗时显著。

**解决方案**：
- 采用批量处理策略，将文档分组后并行构建临时索引，最后合并
- 优化数据结构选择，使用字典推导式替代多次循环操作

#### 6.2.2 tf_idf检索实现时时间过长
## 6.2 遇到的问题与解决方案
### 6.2.1 索引构建效率问题
**问题描述**：原始倒排表构建过程中，文档遍历和索引更新操作频繁，导致整体构建速度较慢，特别是处理接近十万级文档时耗时显著。

**解决方案**：
- 采用批量处理策略，将文档分组后并行构建临时索引，最后合并
- 优化数据结构选择，使用字典推导式替代多次循环操作

### 6.2.2 TF-IDF检索实现时时间过长问题

**问题描述**：TF-IDF检索系统首次查询耗时长达4-5秒，主要瓶颈在于：
- 文档向量构建过程需要遍历所有词项和文档
- 大规模数据集（131,907个词项，90,103个文档）导致计算复杂度高
- 每次初始化都需要重新计算IDF和文档向量

#### 解决方案

##### 1. 延迟加载与缓存优化
```python
def doc_vectors(self):
    """
    延迟加载文档向量，避免不必要的计算
    """
    if self._doc_vectors is None:
        self._doc_vectors = self._build_document_vectors()
    return self._doc_vectors
```

**优化效果**：
- 仅在首次需要时构建文档向量
- 后续查询直接使用缓存结果
- 查询时间从4.7秒降至毫秒级

##### 2. 反向遍历算法优化
```python
def _build_document_vectors(self):
    """
    采用反向遍历策略：对每个词项，只处理包含它的文档
    避免全文档遍历，大幅减少计算量
    """
    doc_vectors = {}
    for doc in self.docs_list:
        doc_vectors[doc] = {}

    # 优化：按词项处理，只处理相关文档
    for token, token_docs in self.inverted_list.items():
        idf = self.idf_values.get(token, 0)
        for doc in token_docs:  # 只遍历包含该词项的文档
            tf = len(token_docs[doc])
            doc_vectors[doc][token] = tf * idf
```
**优化效果**：
- 计算复杂度从O(文档数×词项数)降至O(非零元素数)
#### 双层跳表指针时间内存占比过大问题
- **问题描述：** 双层指针读取时时间到达2-3秒，存储空间比单层指针大2倍。
- **解决方法：** 1.优化指针的保存时的数据结构,将二级指针有字典变为数组,有效减少其占比 变化如下：`{  'type': 'level1','end': end}` 变为`{   'type': 'level1','end': end,'level2': []}'`
   2. 对于读取时间过长：创建类时生成并保存到内存中，有效提高其检索和读取速率
```python
class skip_pointer:
    def __init__(self, inverted_list):
        self.inverted_index = inverted_list
        self.line_number = len(self.inverted_index.keys())
        self.df = None
        self.tokens = sorted(self.inverted_index.keys())
        self.create_skip_pointers()
    def create_skip_pointers(self):
        self.one_skip_list = self.create_one_skip_pointer()
        self.two_skip_list = self.create_two_skip_pointer()
```   


### 6.3 改进方向

#### 6.3.1 文本处理优化
- 实现更高级的文本预处理技术，包括词干提取、词形还原和命名实体识别
- 引入词性标注，支持基于词性的查询过滤

#### 6.3.2 实验扩展
- 增加爬虫实践模块，实现自动化文档采集和预处理
- 探索不同检索模型（如语言模型、BM25）的性能对比


---

## 附录

### 附录A：核心代码结构
```
项目根目录/
├── init.py               # 核心类定义（倒排表、提取待检索文本）
├── search.py             # 检索功能实现（布尔检索、TF-IDF）
├── main.py               # 主程序入口（交互式搜索界面）
├── skip_pointer.py       # 跳表指针实现
├── events_data.json      # 预处理后的文档数据
├── inverted_list.json    # 倒排索引JSON存储
├── stopwords.txt         # 停用词列表
├── fenlei/               # 分类相关数据文件
├── 词典_原始.txt          # 原始词典数据
├── 词典_前端编码.txt      # 压缩后的词典数据
├── 倒排表_原始.csv        # 原始倒排表数据
└── 倒排表_差值编码.csv    # 差值编码压缩的倒排表数据
```


### 附录B：交互式检索系统功能说明

#### 检索选项
1. **单个词项搜索**：支持单个词项的精确匹配，可选择使用单层或双层跳表优化
2. **短语检索**：支持多词短语的连续匹配，保持词项顺序和位置关系
3. **布尔查询**：支持AND/OR/NOT操作符组合，实现复杂逻辑查询
4. **TF-IDF搜索**：基于向量空间模型的相关度排序检索
5. **退出系统**：退出交互式界面

#### 使用示例
```
=== 欢迎使用检索系统===
1. 单个词项搜索
2. 短语检索
3. 布尔查询 (支持 +(OR), *(AND), -(NOT) 操作符)
4. tf_idf 搜索
5. 退出

==================================================
请选择功能 (1-5): 1
请输入要搜索的词项: python
请选择跳表类型: 1(单层)/2(双层): 1
[{'token': 'python', 'file_name': '100263822', 'positions': [3, 23]}...]

```

### 附录C：关键代码片段

#### 1. 字典内存大小计算
```python
def get_dict_size(d):
    """递归计算字典及其所有内容的总大小"""
    size = sys.getsizeof(d)
    for key, value in d.items():
        size += sys.getsizeof(key)
        if isinstance(value, dict):
            size += get_dict_size(value)
        else:
            size += sys.getsizeof(value)
    return size
```

#### 2. 短语检索实现
```python
    def select_texts(self, texts, is_one_skip=True):
        '''短语检索函数，根据select_text返回的位置信息判断是否构成短语
        [{'file_name': 文件名, 'phrase_positions': [(起始位置, 结束位置)]}]
        '''
        start_time = time.time()
        clean = create_token()
        stopwords = clean.read_stopwords()
        tokens = nltk.word_tokenize(texts)
        valid_texts = clean.clean_token(tokens=tokens, stopwords=stopwords)
        print('分词结果：', valid_texts)
        if not valid_texts:
            print('没有有效的搜索文本')
            return [], time.time() - start_time
        if len(valid_texts) == 1:
            result = self.select_text(valid_texts[0], is_one_skip)
            phrase_results = []
            for item in result:
                phrase_positions = []
                for pos in item['positions']:
                    phrase_positions.append((pos, pos))
                phrase_results.append({
                    'file_name': item['file_name'],
                    'phrase_positions': phrase_positions
                })
            return phrase_results, time.time() - start_time

        term_results = {}
        for text in valid_texts:
            result = self.select_text(text, is_one_skip)
            file_positions = {}
            for item in result:
                file_positions[item['file_name']] = item['positions']
            term_results[text] = file_positions

        common_files = set(term_results[valid_texts[0]].keys())
        for text in valid_texts[1:]:
            common_files.intersection_update(term_results[text].keys())
        if not common_files:
            print('没有找到包含所有词项的文件')
            return [], time.time() - start_time
        phrase_matches = []
        for file_name in common_files:
            first_token_positions = term_results[valid_texts[0]][file_name]
            phrase_positions = []
            for start_pos in first_token_positions:
                is_phrase = True
                current_pos = start_pos
                '''...二分法检索...'''

            if phrase_positions:
                phrase_matches.append({
                    'file_name': file_name,
                    'phrase_positions': phrase_positions
                })

        end_time = time.time()
        times = end_time - start_time
        return phrase_matches, times
```

#### 3. TF-IDF向量空间检索
```python
    def search(self, query_text, top_k=10):
        """
        使用TF-IDF和向量空间模型进行文档检索
        """
        start_time = time.time()
        clean = create_token()
        stopwords = clean.read_stopwords()
        tokens = nltk.word_tokenize(query_text)
        query_tokens = clean.clean_token(tokens=tokens, stopwords=stopwords)
        print(f"查询文本: '{query_text}'")
        print(f"清理后的查询词项: {query_tokens}")
        # 如果查询词项为空，直接返回
        if not query_tokens:
            print("查询词项为空，无法执行检索")
            return []
        query_vector = self._create_query_vector(query_tokens)
        relevant_docs = set()
        for token in query_tokens:
            if token in self.inverted_list:
                relevant_docs.update(self.inverted_list[token].keys())
        
        print(f"找到 {len(relevant_docs)} 个包含查询词的相关文档")
        # 计算这些相关文档的相似度
        similarities = []
        for doc in relevant_docs:
            if doc in self.doc_vectors:
                similarity = self._cosine_similarity(query_vector, self.doc_vectors[doc])
                if similarity > 0:
                    similarities.append((doc, similarity))
        #  按相似度排序并返回前k个结果
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_results = similarities[:top_k]
        
        query_time = time.time() - start_time
        print(f"检索到 {len(similarities)} 个相关文档")
        print(f"返回前 {len(top_results)} 个最相关文档:")
        for i, (doc, similarity) in enumerate(top_results, 1):
            print(f"{i}. 文档: {doc}, 相似度: {similarity:.4f}")
        print(f"查询耗时: {query_time:.4f} 秒")
        return top_results
```

#### 4. 交互式搜索界面实现
```python
def interactive_search():
        """交互式搜索界面"""
        skip=skip_pointer(optimizer.load_separate_structures())
        while True:
            print("\n=== 欢迎使用检索系统===")
            print("1. 单个词项搜索")
            print("2. 短语检索")
            print("3. 布尔查询 (支持 +(OR), *(AND), -(NOT) 操作符)")
            print("4. tf_idf 搜索")
            print("5. 退出")

            print("\n" + "="*50)
            choice = input("请选择功能 (1-5): ")

            if choice == '1':
                term = input("请输入要搜索的词项(布尔查询表达式 例如: 'recipe * exchange + food'):")
                skip_choice = input("请选择跳表类型: 1(单层)/2(双层): ")
                if skip_choice == '1':
                    result=skip.select_text(term,is_one_skip=True)
                elif skip_choice == '2':
                    result=skip.select_text(term,is_one_skip=False)
                else:
                    print('flase')
                    result=[]
                print(result)
            elif choice == '2':
                term = input("请输入要搜索的词项(布尔查询表达式 例如: 'recipe * exchange + food'):")
                skip_choice = input("请选择跳表类型: 1(单层)/2(双层): ")
                if skip_choice == '1':
                    print(skip.select_texts(term,is_one_skip=True))
                elif skip_choice == '2':
                    print(skip.select_texts(term,is_one_skip=False))
                else:print('flase')

            elif choice == '3':
                term = input("请输入要搜索的词项(布尔查询表达式 例如: 'recipe * exchange + food'):")
                compress_choice = input("是否使用压缩倒排表? (y/n): ").lower()
                if compress_choice == 'y':
                    complex_search(term,is_compress_invarted=True)
                elif compress_choice == 'n':
                    complex_search(term,is_compress_invarted=False)
                else:
                    print('重新选择')

            elif choice == '4':
                query = input("请输入要搜索的词项: ")
                search_event = tf_idf()
                result=search_event.search(query)
                print(result)
            elif choice == '5':
                print('正在退出....')
                break
            
            else:
                print("无效的选择，请重新输入1-6之间的数字。")
                input("按回车键继续...")
```

