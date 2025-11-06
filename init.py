import math
import string
import time
import random
import pickle
import gzip
import os
from collections import defaultdict
import json
import pandas as pd
import re
from tqdm import tqdm
import nltk
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
    def __init__(self, inverted_list):
        # 直接使用参数传递的倒排表数据，格式为{token: {file_name: [positions]}}
        self.inverted_index = inverted_list
        # 计算token数量
        self.line_number = len(self.inverted_index.keys())
        self.df = None
        # 提取所有token并排序，用于二分查找
        self.tokens = sorted(self.inverted_index.keys())
        # 在初始化时就生成跳表
        self.create_skip_pointers()
    
    def create_skip_pointers(self):
        # 一次性创建并缓存两级跳表
        self.one_skip_list = self.create_one_skip_pointer()
        self.two_skip_list = self.create_two_skip_pointer()
    
    def create_one_skip_pointer(self):
        if not hasattr(self, 'tokens'):
            print('倒排表未加载')
            return
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
                #'level2': None  # 存储二级指针
            }
        return skip_list
    
    def create_two_skip_pointer(self):
        '''二级指针'''
        if not hasattr(self, 'tokens'):
            print('倒排表未加载')
            return
        
        # 优化二级指针结构，使用更高效的存储方式
        skip_list = {}
        # 一级块大小 - 适当调大以减少块数
        level1_block_size = max(1, int(math.sqrt(self.line_number)))
        # 二级块大小 - 基于一级块大小
        level2_block_size = max(1, int(math.sqrt(level1_block_size)))

        for i in range(0, self.line_number, level1_block_size):
            end = min(i + level1_block_size - 1, self.line_number - 1)
            skip_list[i] = {'end': end}
        return skip_list
    
    def select_text(self, text, is_one_skip):
        '''使用跳表指针，从倒排表获得单个token的结果'''
        # 选择使用的跳表
        skip_list = self.one_skip_list if is_one_skip else self.two_skip_list
        # 先检查token是否存在
        if text not in self.inverted_index:
            return []
        
        # 根据是否使用单层指针选择不同的搜索策略
        if is_one_skip:
            # 单层指针搜索
            current = 0
            while current < len(self.tokens):
                if current in skip_list:
                    block_end = skip_list[current]['end']
                    if self.tokens[block_end] < text:
                        current = block_end + 1
                        continue
                    # 在当前块内顺序查找
                    for i in range(current, block_end + 1):
                        if self.tokens[i] == text:
                            return self._format_results(text)
                        elif self.tokens[i] > text:
                            return []
                    return []
                # 如果当前位置不在跳表中，顺序查找
                if self.tokens[current] == text:
                    return self._format_results(text)
                elif self.tokens[current] > text:
                    return []
                current += 1
        else:
            # 双层指针搜索 - 使用优化的二分查找结合跳表
            left, right = 0, len(self.tokens) - 1
            
            # 使用二级跳表进行粗定位
            if hasattr(self, 'two_skip_list'):
                current = 0
                while current < len(self.tokens) and current in self.two_skip_list:
                    block_end = self.two_skip_list[current]['end']
                    if self.tokens[block_end] < text:
                        current = block_end + 1
                    else:
                        left = current
                        right = block_end
                        break
            
            # 在定位的范围内使用二分查找
            while left <= right:
                mid = (left + right) // 2
                if self.tokens[mid] == text:
                    return self._format_results(text)
                elif self.tokens[mid] < text:
                    left = mid + 1
                else:
                    right = mid - 1
        
        return []

    def _format_results(self, text):
        '''格式化查询结果为统一格式，包含位置信息'''
        file_names = self.inverted_index[text]
        if isinstance(file_names, list):

            return [{'token': text, 'file_name': fname, 'positions': []} for fname in file_names]
        elif isinstance(file_names, str):

            return [{'token': text, 'file_name': fname, 'positions': []} for fname in file_names.split(',')]
        elif isinstance(file_names, dict):
            return [{'token': text, 'file_name': fname, 'positions': sorted(positions)} 
                    for fname, positions in file_names.items()]
        return []

    def select_texts(self, texts, is_one_skip=True):
        '''短语检索函数，根据select_text返回的位置信息判断是否构成短语
        
        Args:
            texts: 短语分词列表，如['hello', 'world']
            is_one_skip: 是否使用单层跳表
            
        Returns:
            (包含短语的文件列表, 搜索时间)
            文件列表格式为[{'file_name': 文件名, 'phrase_positions': [(起始位置, 结束位置)]}]
        '''
        start_time = time.time()
        clean=create_token()
        stopwords = clean.read_stopwords()
        tokens = nltk.word_tokenize(texts)
        valid_texts = clean.clean_token(tokens=tokens, stopwords=stopwords)
        print('分词结果：',valid_texts)
        if not valid_texts:
            print('没有有效的搜索文本')
            return [], time.time() - start_time
        
        # 对于单个词项，直接返回select_text的结果
        if len(valid_texts) == 1:
            result = self.select_text(valid_texts[0], is_one_skip)
            # 转换为要求的格式
            phrase_results = []
            for item in result:
                phrase_positions = []
                # 如果有位置信息，每个位置都作为一个短语位置
                for pos in item['positions']:
                    phrase_positions.append((pos, pos))  # (start, end)格式
                phrase_results.append({
                    'file_name': item['file_name'],
                    'phrase_positions': phrase_positions
                })
            return phrase_results, time.time() - start_time
        
        # 获取每个词项的查询结果（包含位置信息）
        term_results = {}
        for text in valid_texts:
            result = self.select_text(text, is_one_skip)
            # 转换为 {file_name: positions} 的格式
            file_positions = {}
            for item in result:
                file_positions[item['file_name']] = item['positions']
            term_results[text] = file_positions
        
        # 找出所有文件的交集
        common_files = set(term_results[valid_texts[0]].keys())
        for text in valid_texts[1:]:
            common_files.intersection_update(term_results[text].keys())
        
        if not common_files:
            print('没有找到包含所有词项的文件')
            return [], time.time() - start_time
        
        # 对于每个共同文件，检查词项是否按顺序连续出现（构成短语）
        phrase_matches = []
        for file_name in common_files:
            # 获取第一个词项的所有位置
            first_token_positions = term_results[valid_texts[0]][file_name]
            phrase_positions = []
            
            # 检查每个可能的起始位置
            for start_pos in first_token_positions:
                is_phrase = True
                current_pos = start_pos
                
                # 检查后续词项是否在正确的位置
                for i in range(1, len(valid_texts)):
                    next_token = valid_texts[i]
                    next_token_positions = term_results[next_token][file_name]
                    
                    # 后续词项应该出现在前一个词项的下一个位置
                    expected_pos = current_pos + 1
                    
                    # 检查expected_pos是否在next_token的位置列表中
                    if expected_pos not in next_token_positions:
                        is_phrase = False
                        break
                    
                    current_pos = expected_pos
                
                # 如果是有效的短语，记录起始和结束位置
                if is_phrase:
                    phrase_positions.append((start_pos, current_pos))
            
            if phrase_positions:
                phrase_matches.append({
                    'file_name': file_name,
                    'phrase_positions': phrase_positions
                })
        
        end_time = time.time()
        times = end_time - start_time
        return phrase_matches, times
    
    def input_participle(self, num_tokens=10):

        # 确保不会选择过多token
        num_tokens = min(num_tokens, len(self.tokens))
        # 随机选择token
        texts = random.sample(self.tokens, num_tokens)
        
        print(f'随机选择的{num_tokens}个查询token：')
        for i, text in enumerate(texts, 1):
            print(f'{i}. {text}')
        
        print('\n单层指针测试：')
        # 单层指针测试
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
        # 双层指针测试
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
# print("跳表索引查询性能测试")
# print("=" * 50)
# skip_pointer = skip_pointer()
# skip_pointer.input_participle(num_tokens=300)

#---------------------倒排表压缩与优化---——————————————————————————————
class optimize_inverted(create_token):
    def __init__(self, json_name='events_data.json'):
        super().__init__(json_name, tokenizer_path=None)

    def inverted_list(self):
        """
        倒排表格式
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
    
    def front_coding(self, tokens):
        """
        对排序后的词项列表进行前端编码压缩
        """
        if not tokens:
            return []
        
        encoded_dict = []
        i = 0
        block_size = 8  # 更大的块大小以提高压缩率
        
        while i < len(tokens):
            # 取当前块的第一个词作为前缀
            prefix = tokens[i]
            # 第一个词特殊标记为-1表示完整词项
            encoded_dict.append((-1, prefix))
            
            # 处理当前块的后续词项
            j = 1
            while i + j < len(tokens) and j < block_size:
                current_token = tokens[i + j]
                # 计算公共前缀长度
                common_len = 0
                min_len = min(len(prefix), len(current_token))
                while common_len < min_len and prefix[common_len] == current_token[common_len]:
                    common_len += 1
                # 存储公共长度和后缀
                suffix = current_token[common_len:]
                encoded_dict.append((common_len, suffix))
                j += 1
            
            i += j
        
        return encoded_dict
    
    def front_decoding(self, encoded_dict):
        """
        对前端编码的词典进行解码
        """
        tokens = []
        prefix = None
        
        for item in encoded_dict:
            common_len, data = item
            if common_len == -1:
                # -1表示完整词项，作为新块的前缀
                prefix = data
                tokens.append(prefix)
            else:
                # 根据公共前缀长度和后缀重建词项
                if prefix is not None:
                    full_token = prefix[:common_len] + data
                    tokens.append(full_token)
        
        return tokens
    
    def delta_encoding(self, positions):
        """
        对位置列表进行差值编码
        """
        if not positions:
            return []
        
        encoded = [positions[0]]  # 第一个元素保持不变
        for i in range(1, len(positions)):
            encoded.append(positions[i] - positions[i-1])  # 存储差值
        
        return encoded
    
    def delta_decoding(self, encoded_positions):
        """
        对差值编码的位置列表进行解码
        """
        if not encoded_positions:
            return []
        
        decoded = [encoded_positions[0]]  # 第一个元素保持不变
        for i in range(1, len(encoded_positions)):
            decoded.append(decoded[i-1] + encoded_positions[i])  # 累加差值得到原始位置
        
        return decoded
    
    def save_separate_structures(self, inverted_index):
        """
        分离存储词典和倒排表
        - 词典使用前端编码保存为txt文件
        - 倒排表使用差值编码保存为CSV文件
        - 同时保存压缩前后的词典和倒排表用于比较
        返回:
        - 是否保存成功
        """
        try:
            # 使用汉语命名的文件
            dict_file = "词典_前端编码.txt"  # 压缩后的词典
            original_dict_file = "词典_原始.txt"  # 原始词典
            postings_file = "倒排表_差值编码.csv"  # 压缩后的倒排表
            original_postings_file = "倒排表_原始.csv"  # 原始倒排表
            
            # 1. 处理词典 - 提取并排序所有词项
            sorted_tokens = sorted(inverted_index.keys())
            
            # 保存原始词典
            with open(original_dict_file, 'w', encoding='utf-8') as f:
                for token in sorted_tokens:
                    f.write(f"{token}\n")
            
            # 前端编码压缩词典
            encoded_dict = self.front_coding(sorted_tokens)
            
            # 保存压缩后的词典（简化格式）
            with open(dict_file, 'w', encoding='utf-8') as f:
                for common_len, data in encoded_dict:
                    # -1表示完整词项，其他值表示公共前缀长度
                    f.write(f"{common_len}:{data}\n")
            
            # 2. 处理倒排表 - 保存原始倒排表
            with open(original_postings_file, 'w', encoding='utf-8') as f:
                f.write("token,file_name,positions\n")
                for token, file_positions in inverted_index.items():
                    for file_name, positions in file_positions.items():
                        positions_str = ','.join(map(str, positions))
                        f.write(f"{token},{file_name},{positions_str}\n")
            
            # 3. 使用差值编码保存倒排表
            with open(postings_file, 'w', encoding='utf-8') as f:
                # 写入表头
                f.write("token_id,file_name,encoded_positions\n")
                
                for token_id, token in enumerate(sorted_tokens):
                    # 使用token_id作为指针，不存储token本身
                    for file_name, positions in inverted_index[token].items():
                        # 差值编码位置列表
                        encoded_positions = self.delta_encoding(positions)
                        # 将编码后的位置转换为字符串
                        positions_str = ','.join(map(str, encoded_positions))
                        f.write(f"{token_id},{file_name},{positions_str}\n")
            
            # 计算压缩率
            dict_size = os.path.getsize(dict_file)
            original_dict_size = os.path.getsize(original_dict_file)
            postings_size = os.path.getsize(postings_file)
            original_postings_size = os.path.getsize(original_postings_file)
            
            # 计算词典压缩率
            dict_compression_ratio = (1 - dict_size / original_dict_size) * 100 if original_dict_size > 0 else 0
            # 计算倒排表压缩率
            postings_compression_ratio = (1 - postings_size / original_postings_size) * 100 if original_postings_size > 0 else 0
            # 计算总体压缩率
            original_total_size = original_dict_size + original_postings_size
            compressed_total_size = dict_size + postings_size
            total_compression_ratio = (1 - compressed_total_size / original_total_size) * 100 if original_total_size > 0 else 0
            
            print("\n--- 词典压缩结果 ---")
            print(f"原始词典文件已保存: {original_dict_file} ({original_dict_size / 1024:.2f} KB)")
            print(f"压缩后词典文件已保存: {dict_file} ({dict_size / 1024:.2f} KB)")
            print(f"词典压缩率: {dict_compression_ratio:.2f}%")
            print("\n--- 倒排表压缩结果 ---")
            print(f"原始倒排表文件已保存: {original_postings_file} ({original_postings_size / 1024:.2f} KB)")
            print(f"压缩后倒排表文件已保存: {postings_file} ({postings_size / 1024:.2f} KB)")
            print(f"倒排表压缩率: {postings_compression_ratio:.2f}%")
            print(f"总压缩率: {total_compression_ratio:.2f}%")
            
            return True
        except Exception as e:
            print(f"保存分离结构时出错: {e}")
            return False
    
    def load_separate_structures(self):
        """
        从分离的文件中加载词典和倒排表，重建完整的倒排索引
        确保返回格式严格为{token: {file_name: [positions]}}
        """
        try:
            dict_file = "词典_前端编码.txt"
            postings_file = "倒排表_差值编码.csv"
            encoded_dict = []
            
            with open(dict_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        common_len_str, data = line.split(':', 1)
                        common_len = int(common_len_str)
                        encoded_dict.append((common_len, data))
                    except (ValueError, IndexError):
                        continue
            sorted_tokens = self.front_decoding(encoded_dict)
            inverted_index = {}  # {token: {file_name: [positions]}}
            
            with open(postings_file, 'r', encoding='utf-8') as f:
                header = next(f)  # 跳过表头
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split(',', 2)
                    if len(parts) != 3:
                        continue
                    
                    token_id = int(parts[0])
                    file_name = parts[1]
                    positions_str = parts[2]

                    # 验证token_id的有效性
                    if token_id < 0 or token_id >= len(sorted_tokens):
                        print(f"无效的token_id: {token_id}")
                        continue
                    token = sorted_tokens[token_id]

                    # 确保位置列表正确解码
                    try:
                        encoded_positions = list(map(int, positions_str.split(',')))
                        positions = self.delta_decoding(encoded_positions)
                        
                        # 确保返回格式严格为{token: {file_name: [positions]}}
                        if token not in inverted_index:
                            inverted_index[token] = {}
                        inverted_index[token][file_name] = positions
                    except Exception as e:
                        print(f"处理位置数据时出错: {e}")
                        continue
            
            print(f"成功加载倒排索引: {len(inverted_index)} 个词项")
            return inverted_index
        except FileNotFoundError as e:
            print(f"文件未找到: {e}")
            return None
        except Exception as e:
            print(f"加载分离结构时出错: {e}")
            return None
    
    def is_ture(self, original_index, loaded_index):
        """
        验证原始倒排索引和加载后的倒排索引是否完全一致
        """
        new_inversted=loaded_index
        old_inverted = original_index
        if new_inversted != old_inverted:
            print("出错")
            return False
        print("---读取倒排表-----")
        # 对于字典类型，使用items()方法获取前10个键值对
        print(list(new_inversted.items())[:3])
        print("-----原倒排表-----")
        print(list(old_inverted.items())[:3])
        return True
# 测试代码
if __name__ == "__main__":
    # 创建优化器实例
    optimizer = optimize_inverted()
    
    # 生成倒排索引
    print("正在生成倒排索引...")
    inverted_index = optimizer.inverted_list()
    print(f"倒排索引生成完成，包含 {len(inverted_index)} 个词项")
    
    # 保存分离的结构
    print("\n正在保存分离的词典和倒排表...")
    optimizer.save_separate_structures(inverted_index)
    
    # 测试加载分离存储的数据
    print("\n正在从分离文件加载倒排索引...")
    loaded_index = optimizer.load_separate_structures()
    
    # 使用is_ture验证函数验证一致性
    if loaded_index:
        print("\n正在验证索引一致性...")
        is_consistent = optimizer.is_ture(inverted_index, loaded_index)
        print(f"一致性验证结果: {'通过' if is_consistent else '失败'}")
    skip=skip_pointer(inverted_list=loaded_index)
    one=skip.select_text(text="recipe",is_one_skip=True)
    skip2=skip_pointer(inverted_list=inverted_index)
    two=skip2.select_text(text="recipe",is_one_skip=True)
    if one==two:
        print("完全相同")
    else :
        print("不一样")


            

