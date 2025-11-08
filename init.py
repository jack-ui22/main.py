import string
import os
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
        pattern = r'<member><member_id>(.*?)</member_id><name>(.*?)</name></member>'
        matches = re.findall(pattern, data)
        result = [{"id": match[0],"name":match[1]} for match in matches]

        return result

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
                lines = f.readlines()
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

        for token in tokens:
            token = token.lower()
            token = re.sub(r'^\W+|\W+$', '', token)
            token = re.sub(r'www.*?', '', token)
            token = re.sub(r'\d+', '', token)
            token = re.sub(r'.*?/.*?', '', token)
            token = re.sub(r'.*?.org.*?', '', token)
            token = re.sub(r'.*?/.*?', '', token)
            if not token:
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
            token_map = {}
            for item in inverted_list:
                token = item['token']
                file_name = item['file_name']
                if token not in token_map:
                    token_map[token] = set()
                token_map[token].add(file_name)

            json_data = {}
            for token, file_set in token_map.items():
                json_data[token] = sorted(file_set)
            with open(output_file, 'w', encoding='utf-8') as jsonfile:
                json.dump(
                    json_data,
                    jsonfile,
                    ensure_ascii=False,
                    indent=4,
                    sort_keys=True
                )

            print(f"倒排索引已成功写入: {output_file}")
            return True

        except Exception as e:
            print(f"写入JSON时出错: {e}")
            return False

# token=create_token()
# inverted=token.inverted_list()
# token.write_to_json(inverted,"inverted_list.json")

#---------------------倒排表压缩与优化---——————————————————————————————
class optimize_inverted(create_token):
    def __init__(self, json_name='events_data.json'):
        super().__init__(json_name, tokenizer_path=None)
        self.inverted_list =self.load_separate_structures()


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
    
    def delta_encoding(self, positions):
        """
        对位置列表进行差值编码
        """
        if not positions:
            return []
        
        encoded = [positions[0]]
        for i in range(1, len(positions)):
            encoded.append(positions[i] - positions[i-1])
        return encoded
    
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
    
    def save_separate_structures(self, inverted_index):
        """
        分离存储词典和倒排表
        - 词典使用前端编码保存为txt文件
        - 倒排表使用差值编码保存为CSV文件
        - 同时保存压缩前后的词典和倒排表用于比较
        """
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

            dict_size = os.path.getsize(dict_file)
            original_dict_size = os.path.getsize(original_dict_file)
            postings_size = os.path.getsize(postings_file)
            original_postings_size = os.path.getsize(original_postings_file)

            dict_compression_ratio = (1 - dict_size / original_dict_size) * 100 if original_dict_size > 0 else 0
            postings_compression_ratio = (1 - postings_size / original_postings_size) * 100 if original_postings_size > 0 else 0
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
                header = next(f)
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

                    try:
                        encoded_positions = list(map(int, positions_str.split(',')))
                        positions = self.delta_decoding(encoded_positions)
                        
                        # {token: {file_name: [positions]}}
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
        return True




            

