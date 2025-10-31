import math
import os
import re
import pandas as pd
from tqdm import tqdm
import nltk
import string
from collections import defaultdict
import time
import json
import base64
import struct
import random
#--------------文件提取-----------------
class xml_analyse:
    def __init__(self, xml_path='data/All_Unpack'):
        self.xml_path = os.path.join(os.getcwd(), xml_path)

    def xml_name(self):
        event_files = []
        ev_files = []
        member_files = []
        group_files = []
        with os.scandir(self.xml_path) as entries:
            for entry in entries:
                if entry.is_file():
                    name = entry.name
                    if re.search(r'RSVPs .*?.xml', name):
                        event_files.append(name)
                    elif re.search(r'PastEvent .*?.xml', name):
                        ev_files.append(name)
                    elif re.search(r'Memeber .*?.xml', name):
                        member_files.append(name)
                    else:
                        group_files.append(name)
        if event_files:
            pd.DataFrame(event_files, columns=['filename']).to_csv("./fenlei/event.csv", index=False)
        if ev_files:
            pd.DataFrame(ev_files, columns=['filename']).to_csv("./fenlei/EV.csv", index=False)
        if member_files:
            pd.DataFrame(member_files, columns=['filename']).to_csv("./fenlei/members.csv", index=False)
        if group_files:
            pd.DataFrame(group_files, columns=['filename']).to_csv("./fenlei/group.csv", index=False)
        print('已完成')

    def events_info(self):
        data = pd.read_csv(os.path.join(os.getcwd(), './fenlei/EV.csv'))
        file_list = data['filename']
        all_events_data = []
        print("正在处理原始文档...")

        for i in tqdm(range(len(file_list))):
            file_name = file_list[i]
            file_path = self.xml_path + f"/{file_name}"

            with open(file_path, 'r', encoding='utf-8') as f:
                xml_content = f.read()

            # 修正：移除值为None的"member"键
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
            # 修正：在循环外处理member信息
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
    def members_info(self,id):
        file_path = os.path.join(os.getcwd(), f"./data/All_Unpack/RSVPs {id}.xml")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = f.read()
        except FileNotFoundError:
            return {}
        # 单次匹配提取所有条目
        pattern = r'<member><member_id>(.*?)</member_id><name>(.*?)</name></member>'
        matches = re.findall(pattern, data)
        # 转换为字典列表
        result = [{"id": match[0],"name":match[1]} for match in matches]

        return result

    def clear_xml(self, text):
        if not text:
            return ""
        # 1. 移除 XML 标签
        text = re.sub(r'<[^>]+>', '', text)
        # 2. 替换 XML 实体转义字符
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
        # 4. 移除 XML 处理指令
        # text = re.sub(r'.*?<br \\/>', '', text)
        text = re.sub(r'<\?.*?\?>', '', text, flags=re.DOTALL)
        # 6. 清理多余空格和换行
        text = re.sub(r'\s+', ' ', text)  # 多个空格替换为单个空格
        text = text.strip()  # 移除首尾空格
        return text


    def group_info(self):

        pass
    def join_info(self):

        pass
# events= xml_analyse()
# events_info = events.events_info()
#-------------分词(需要改进)------------------------
class create_token:
    def __init__(self, json_name='events_data.json',tokenizer_path=None):
        self.json_name = json_name
        self.tokenizer_path = tokenizer_path
        self.inverted_index="inverted.csv"
    def download_token(self):
            nltk.download('punkt_tab')
    def read_stopwords(self):
        stopwords_set = set()
        filename = 'stopwords.txt'
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                lines = f.readlines()  # 一次性读取所有行
                total_lines = len(lines)

            for line in tqdm(lines, total=total_lines, desc="加载停用词"):
                cleaned_line = line.strip()
                if cleaned_line:
                    stopwords_set.add(cleaned_line)
            return stopwords_set
        except FileNotFoundError:
            print(f"错误：停用词文件 '{filename}' 未找到。")
            return set()
        except Exception as e:
            print(f"读取停用词文件时发生错误: {e}")
            return set()

    def inverted_list(self):
        try:
            inverted = {
                'token': None,
                'file_name': None
            }
            inverted_list = []
            df = pd.read_json(self.json_name)
            #self.download_token()
            stopwords = self.read_stopwords()
            key_words = df['key_word'].tolist()
            decs_main = df['description'].tolist()
            for i in tqdm(range(len(key_words))):
                key_word = key_words[i]
                dec_main = decs_main[i]
                if dec_main==None:
                    continue
                # 简化keyword
                finall_key_word = re.findall(
                    r"PastEvent (?P<name>.*?).xml", key_word
                )
                # 对dec_main进行分词并处理
                tokens = nltk.word_tokenize(dec_main)
                filtered_tokens = self.clean_token(tokens=tokens,stopwords=stopwords)
                # 构建倒排记录
                for token in set(filtered_tokens):  # 使用set去重
                    record = inverted.copy()
                    record['token'] = token
                    record['file_name'] = finall_key_word[0] if finall_key_word else "Unknown"
                    inverted_list.append(record)
            return inverted_list

        except FileNotFoundError:
            print('json not found')
            return set()
        except Exception as e:
            print(f"error in read json:{e}")
            return set()

    def clean_token(self, tokens, stopwords):
        cleaned_tokens = []
        puncts = set(string.punctuation)
        # 综合清理正则
        invalid_pattern = re.compile(
            r'^(?:'  # 开始分组
            r'[-.]{2,}|'  # 连续特殊字符
            r'[-./:]{1,2}\w+|'  # 特殊字符开头的词
            r'\w*[-./:]\w*|'  
            r'\w{1,3}|' 
            r'[^a-zA-Z].*' 
            r')$'  # 结束分组
        )
        for token in tokens:
            token = token.lower()
            token = re.sub(r'^\W+|\W+$', '', token)
            # 使用正则过滤不以字母开头的词
            if invalid_pattern.match(token):
                continue
            # 其他清理规则保持不变
            token = re.sub(r'www.*?', '', token)
            token = re.sub(r'\d+', '', token)
            token = re.sub(r'.*?/.*?', '', token)
            token = re.sub(r'.*?.org.*?', '', token)
            token = re.sub(r'.*?/.*?', '', token)
            if not token:  # 确保 token 非空
                continue
            if not token[0].isalpha() and token[0] != '-':
                continue
            if token[0] == '-' and (len(token) < 2 or not token[1].isalpha()):
                continue
            if token in puncts:
                continue
            if token in stopwords or len(token) < 4:
                continue
            cleaned_tokens.append(token)
        return cleaned_tokens

    def write_to_json(self, inverted_list,output_file):
        try:
            # 创建 token 到文件集合的映射
            token_map = {}
            for item in inverted_list:
                token = item['token']
                file_name = item['file_name']

                if token not in token_map:
                    token_map[token] = set()
                token_map[token].add(file_name)

            # 准备 JSON 数据结构
            json_data = {}
            for token, file_set in token_map.items():
                # 将文件集合转换为排序后的列表
                json_data[token] = sorted(file_set)

            # 写入 JSON 文件

            with open(output_file, 'w', encoding='utf-8') as jsonfile:
                json.dump(
                    json_data,
                    jsonfile,
                    ensure_ascii=False,
                    indent=4,  # 格式化输出，提高可读性
                    sort_keys=True  # 按键名（token）排序
                )

            print(f"倒排索引已成功写入: {output_file}")
            return True

        except Exception as e:
            print(f"写入JSON时出错: {e}")
            return False



# #----------------------跳表指针------------------------------
class skip_pointer:
    def __init__(self, inverted_list='inverted_list.json'):
        self.inverted_list = inverted_list
        self.line_number = 0
        self.df=None
        # 在初始化时就生成跳表
        self.read_inverted_list()
        self.create_skip_pointers()

    def read_inverted_list(self):
        # 直接读取JSON文件并转换为字典
        with open(self.inverted_list, 'r', encoding='utf-8') as f:
            self.inverted_index = json.load(f)
        # 提取所有token并排序，用于二分查找
        self.tokens = sorted(self.inverted_index.keys())
        self.line_number = len(self.tokens)
        return self.line_number, self.inverted_index
    
    def create_skip_pointers(self):
        # 一次性创建并缓存两级跳表
        self.one_skip_list = self.create_one_skip_pointer()
        self.two_skip_list = self.create_two_skip_pointer()
    def create_one_skip_pointer(self):
        if not hasattr(self, 'tokens'):
            self.read_inverted_list()
        # 一级指针
        skip_list={}
        block_size = max(1, int(math.sqrt(self.line_number)))
        i = 0
        while i < self.line_number:
            start = i
            i += block_size
            end = min(i, self.line_number) - 1  # 结束位置索引
            # 保存一级指针
            skip_list[start] = {
                'type': 'level1',
                'end': end,
                'level2': None  # 存储二级指针
            }
        return skip_list
    def create_two_skip_pointer(self):
        '''二级指针 - 简化优化版本'''
        if not hasattr(self, 'tokens'):
            self.read_inverted_list()
        
        # 优化二级指针结构，使用更高效的存储方式
        skip_list = {}
        
        # 一级块大小 - 适当调大以减少块数
        level1_block_size = max(1, int(math.sqrt(self.line_number)))
        # 二级块大小 - 基于一级块大小
        level2_block_size = max(1, int(math.sqrt(level1_block_size)))
        
        # 创建简化的双层跳表结构
        # 1. 直接存储每个位置可能的跳跃目标，而不是复杂的嵌套结构
        # 2. 预计算每个位置的最佳跳跃位置
        
        # 首先创建一级指针
        for i in range(0, self.line_number, level1_block_size):
            end = min(i + level1_block_size - 1, self.line_number - 1)
            skip_list[i] = {'end': end}
        
        # 然后为每个位置添加最近的二级指针
        # 为了提高效率，我们只存储直接的跳跃目标，而不是复杂的结构
        # 预先计算每个位置的最佳跳跃位置
        return skip_list
    
    def select_text(self, text, is_one_skip):
        '''使用跳表指针，从倒排表获得单个token的结果 - 优化版本'''
        # 先检查token是否存在
        if text not in self.inverted_index:
            return []
        
        # 根据是否使用单层指针选择不同的搜索策略
        if is_one_skip:
            # 单层指针搜索
            # 直接遍历跳表块进行快速定位
            left, right = 0, len(self.tokens) - 1
            if hasattr(self, 'one_skip_list'):
                current = 0
                # 使用一级指针进行快速定位
                while current < len(self.tokens) and current in self.one_skip_list:
                    block_end = self.one_skip_list[current]['end']
                    if self.tokens[block_end] < text:
                        current = block_end + 1
                    else:
                        # 找到可能包含目标的块
                        left = current
                        right = block_end
                        break
            
            # 在定位的块内进行顺序查找（保持单层指针的特性）
            for i in range(left, right + 1):
                if self.tokens[i] == text:
                    # 找到目标token，返回结果
                    file_names = self.inverted_index[text]
                    if isinstance(file_names, list):
                        return [{'token': text, 'file_name': fname} for fname in file_names]
                    elif isinstance(file_names, str):
                        return [{'token': text, 'file_name': fname} for fname in file_names.split(',')]
                    elif isinstance(file_names, dict):
                        return [{'token': text, 'file_name': fname} for fname in file_names.keys()]
                elif self.tokens[i] > text:
                    break  # 因为是排序数组，后面的元素更大，提前退出
        else:
            # 双层指针搜索 - 使用优化的二分查找结合跳表
            left, right = 0, len(self.tokens) - 1
            
            # 使用一级块进行粗定位
            if hasattr(self, 'two_skip_list'):
                current = 0
                while current < len(self.tokens) and current in self.two_skip_list:
                    block_end = self.two_skip_list[current]['end']
                    if self.tokens[block_end] < text:
                        current = block_end + 1
                    else:
                        # 找到可能包含目标的块，缩小搜索范围
                        left = current
                        right = block_end
                        break
            
            # 在定位的范围内使用二分查找
            while left <= right:
                mid = (left + right) // 2
                if self.tokens[mid] == text:
                    # 找到目标token，返回结果
                    file_names = self.inverted_index[text]
                    if isinstance(file_names, list):
                        return [{'token': text, 'file_name': fname} for fname in file_names]
                    elif isinstance(file_names, str):
                        return [{'token': text, 'file_name': fname} for fname in file_names.split(',')]
                    elif isinstance(file_names, dict):
                        return [{'token': text, 'file_name': fname} for fname in file_names.keys()]
                elif self.tokens[mid] < text:
                    left = mid + 1
                else:
                    right = mid - 1
        
        return []

    def select_texts(self, texts, is_one_skip):
        '''result为二维列表，该函数返回多个列表，返回包含最接近的结果，
        如有三个分词，如没有同时拥有的，那么返回包含两个结果，以此类推'''
        start_time = time.time()
        # 优化：先过滤空文本
        valid_texts = [text for text in texts if text.strip()]
        if not valid_texts:
            print('没有有效的搜索文本')
            return [], time.time() - start_time
        
        results = []
        for text in valid_texts:
            result = self.select_text(text, is_one_skip)
            results.append(result)
        
        # 统计每个文件出现的次数
        file_count = defaultdict(int)
        for i in range(len(results)):  # 移除tqdm以减少开销，除非处理大量数据
            # 使用集合去重，避免同一分词结果中重复计数
            unique_files = set()
            for item in results[i]:
                unique_files.add(item['file_name'])
            # 更新文件计数
            for file_name in unique_files:
                file_count[file_name] += 1
        
        # 按共同出现次数分组文件
        count_groups = defaultdict(list)
        for file_name, count in file_count.items():
            count_groups[count].append(file_name)
        
        # 从高到低排序
        sorted_counts = sorted(count_groups.keys(), reverse=True)
        
        # 添加空列表检查
        if not sorted_counts:  # 检查列表是否为空
            print('没有找到匹配的文件')
            end_time = time.time()
            times = end_time - start_time
            return [], times
        
        max_count = sorted_counts[0]
        print('出现次数:', max_count)
        # 优化：按相关性排序结果
        final_results = []
        for count in sorted_counts:
            # 只返回与最高频率相差不超过1的结果
            if max_count - count <= 1:
                final_results.append(sorted(count_groups[count]))
            else:
                break
        
        end_time = time.time()
        times = end_time - start_time
        return final_results, times
    def select_text(self, text, is_one_skip):
        '''使用跳表指针，从倒排表获得单个token的结果'''
        # 选择使用的跳表
        skip_list = self.one_skip_list if is_one_skip else self.two_skip_list
        
        # 先检查token是否存在
        if text not in self.inverted_index:
            return []
        
        current = 0
        results = []
        
        # 使用跳表进行查找
        while current < len(self.tokens):
            current_token = self.tokens[current]
            
            # 找到目标token
            if current_token == text:
                file_names = self.inverted_index[current_token]
                # 处理不同格式的file_names
                if isinstance(file_names, list):
                    results = [{'token': text, 'file_name': fname} for fname in file_names]
                elif isinstance(file_names, str):
                    results = [{'token': text, 'file_name': fname} for fname in file_names.split(',')]
                elif isinstance(file_names, dict):
                    results = [{'token': text, 'file_name': fname} for fname in file_names.keys()]
                break
            
            # 如果当前token已经大于目标text，说明目标不存在
            if current_token > text:
                break
            
            # 跳表指针处理 - 优化版本
            if current in skip_list:
                skip_info = skip_list[current]
                
                # 首先尝试使用二级指针（如果存在）
                if not is_one_skip and 'level2' in skip_info and skip_info['level2']:
                    # 找到当前位置之后最近的二级指针起始位置
                    # 这里不再排序，直接使用字典的特性
                    best_jump = current
                    # 快速检查：如果当前块的末尾都小于目标，则直接跳到块末尾
                    if skip_info['end'] < len(self.tokens) and self.tokens[skip_info['end']] <= text:
                        current = skip_info['end']
                        continue
                    
                    # 只检查最接近当前位置的几个二级指针
                    # 获取所有起始位置大于current的二级指针
                    level2_positions = [pos for pos in skip_info['level2'].keys() if pos > current]
                    if level2_positions:
                        # 找到第一个超过text的位置
                        for pos in sorted(level2_positions):
                            end_pos = skip_info['level2'][pos]
                            if end_pos < len(self.tokens) and self.tokens[end_pos] <= text:
                                best_jump = end_pos
                            else:
                                break
                        if best_jump > current:
                            current = best_jump
                            continue
                
                # 使用一级指针
                if 'end' in skip_info:
                    skip_end = skip_info['end']
                    # 确保索引有效
                    if skip_end < len(self.tokens) and self.tokens[skip_end] <= text:
                        current = skip_end
                        continue
            
            # 如果没有可以跳的指针，就顺序查找
            current += 1
        
        return results

    def input_participle(self, num_tokens=10):
        # 从倒排索引的tokens中随机选择指定数量的token（增加查询数量）

        # 确保不会选择过多token
        num_tokens = min(num_tokens, len(self.tokens))
        # 随机选择token
        texts = random.sample(self.tokens, num_tokens)
        
        print(f'随机选择的{num_tokens}个查询token：')
        for i, text in enumerate(texts, 1):
            print(f'{i}. {text}')
        
        print('\n单层指针测试：')
        # 单层指针测试 - 单次查询（不重复）
        start_time = time.time()
        all_results1 = []
        for text in texts:
            result = self.select_text(text, is_one_skip=True)
            all_results1.append(result)
        total_time1 = time.time() - start_time
        avg_time1 = total_time1 / num_tokens
        
        # 输出单层指针的时间信息
        print(f'总查询时间：{total_time1:.6f}秒')
        print(f'平均每个查询时间：{avg_time1:.6f}秒')
        
        # 输出部分结果示例
        print('\n查询结果示例：')
        for i, (text, result) in enumerate(zip(texts[:3], all_results1[:3]), 1):  # 只显示前3个结果
            if result:
                file_count = len(set(item['file_name'] for item in result))
                print(f'{i}. 查询 "{text}" 找到 {file_count} 个相关文件')
                # 显示前2个文件
                for j, item in enumerate(result[:2], 1):
                    print(f'   - {item["file_name"]}')
                if len(result) > 2:
                    print(f'   ... 等{len(result)-2}个文件')
            else:
                print(f'{i}. 查询 "{text}" 未找到结果')
        
        print('\n' + '-'*50 + '\n')
        print('双层指针测试：')
        # 双层指针测试 - 单次查询（不重复）
        start_time2 = time.time()
        all_results2 = []
        for text in texts:
            result = self.select_text(text, is_one_skip=False)
            all_results2.append(result)
        total_time2 = time.time() - start_time2
        avg_time2 = total_time2 / num_tokens
        
        # 输出双层指针的时间信息
        print(f'总查询时间：{total_time2:.6f}秒')
        print(f'平均每个查询时间：{avg_time2:.6f}秒')
        
        # 输出性能比较
        if total_time1 > total_time2:
            speedup = total_time1 / total_time2
            print(f'\n双层指针比单层指针快 {speedup:.2f} 倍')
        else:
            slowdown = total_time2 / total_time1
            print(f'\n单层指针比双层指针快 {slowdown:.2f} 倍')

# 运行单次随机查询测试
print("跳表索引查询性能测试")
print("=" * 50)
skip_pointer = skip_pointer()
skip_pointer.input_participle(num_tokens=300)  # 增加到15个随机查询token

#---------------------倒排表压缩与优化---——————————————————————————————
class optimize_inverted(create_token):
    def __init__(self, json_name='events_data.json'):
        super().__init__(json_name, tokenizer_path=None)

    def inverted_list(self):
        """
        {token: {file_name: [positions]}}
        """
        try:
            # 使用高效的数据结构：双层字典
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
                # 提取文件名
                finall_key_word = re.findall(
                    r"PastEvent (?P<name>.*?).xml", key_word
                )
                file_name = finall_key_word[0] if finall_key_word else "Unknown"
                # 分词和处理
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

    def write_to_json(self, inverted_index, output_file):
        try:
            # 准备JSON数据结构
            json_data = {}
            for token, file_positions in inverted_index.items():
                # 直接使用位置列表（整数列表）作为值
                json_data[token] = {
                    file_name: positions  # 保持位置为整数列表
                    for file_name, positions in file_positions.items()
                }

            # 写入JSON文件
            with open(output_file, 'w', encoding='utf-8') as jsonfile:
                json.dump(
                    json_data,
                    jsonfile,
                    ensure_ascii=False,
                    indent=4,  # 使用缩进格式化，提高可读性
                    sort_keys=True  # 按键名（token）排序
                )
            print(f"优化后的倒排索引已成功写入: {output_file}")
            return True

        except Exception as e:
            print(f"写入JSON时出错: {e}")
            return False

    def compress_list(self, inverted_index):
        """
        该方法通过将位置列表转换为字节格式，并进行Base64编码，来压缩倒排索引。
        """
        compressed_index = defaultdict(dict)
        for token, file_positions in inverted_index.items():
            for file_name, positions in file_positions.items():
                # 使用 struct 将位置列表转换为字节格式
                # 通过 struct.pack() 将整数列表转换为字节串
                byte_positions = struct.pack(f'{len(positions)}i', *positions)

                # 使用 Base64 对字节串进行编码
                encoded_positions = base64.b64encode(byte_positions).decode('utf-8')
                compressed_index[token][file_name] = encoded_positions

        return dict(compressed_index)

    def block_storage(self, inverted_index, block_size=1000, output_dir='blocks'):
        os.makedirs(output_dir, exist_ok=True)
        blocks = defaultdict(dict)
        current_block = 0
        current_size = 0

        for token, file_positions in inverted_index.items():
            for file_name, positions in file_positions.items():
                if current_size >= block_size:
                    with open(f"{output_dir}/block_{current_block}.json", 'w', encoding='utf-8') as jsonfile:
                        json.dump(blocks, jsonfile, ensure_ascii=False, indent=4, sort_keys=True)
                    current_block += 1
                    blocks = defaultdict(dict)
                    current_size = 0

                blocks[token][file_name] = positions
                current_size += 1
        if blocks:
            with open(f"{output_dir}/block_{current_block}.json", 'w', encoding='utf-8') as jsonfile:
                json.dump(blocks, jsonfile, ensure_ascii=False, indent=4, sort_keys=True)

        print(f"倒排索引已成功按块存储到: {output_dir}")
        return True


# token = optimize_inverted()
# inverted_index = token.inverted_list()
#
# token.write_to_json(inverted_index, 'inverted_list.json')
#
# compressed_index = token.compress_list(inverted_index)
# token.write_to_json(compressed_index, 'compressed.json')
#
# token.block_storage(inverted_index, block_size=500, output_dir='inverted_blocks')