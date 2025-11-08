import time
import math
from init import optimize_inverted,create_token
import nltk
import random
import sys


#----------------------跳表指针------------------------------
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
    def select_text(self, text, is_one_skip):
        '''使用跳表指针，从倒排表获得单个token的结果'''
        skip_list = self.one_skip_list if is_one_skip else self.two_skip_list
        if text not in self.inverted_index:
            return []
        if is_one_skip:
            current = 0
            while current < len(self.tokens):
                if current in skip_list:
                    block_end = skip_list[current]['end']
                    if self.tokens[block_end] < text:
                        current = block_end + 1
                        continue
                    for i in range(current, block_end + 1):
                        if self.tokens[i] == text:
                            return self._format_results(text)
                        elif self.tokens[i] > text:
                            return []
                    return []
                if self.tokens[current] == text:
                    return self._format_results(text)
                elif self.tokens[current] > text:
                    return []
                current += 1
        else:

            level1_positions = sorted(skip_list.keys())
            current_index = 0
            
            while current_index < len(level1_positions):

                current_block_start = level1_positions[current_index]
                current_block = skip_list[current_block_start]
                current_block_end = current_block['end']

                if self.tokens[current_block_end] < text:

                    if current_block['level2']:
                        max_jump_index = current_index
                        for jump_pos in current_block['level2']:
                            if jump_pos in skip_list:
                                jump_block_end = skip_list[jump_pos]['end']
                                if self.tokens[jump_block_end] < text:
                                    try:
                                        jump_index = level1_positions.index(jump_pos)
                                        max_jump_index = max(max_jump_index, jump_index)
                                    except ValueError:
                                        pass
                        if max_jump_index > current_index:
                            current_index = max_jump_index
                        else:
                            current_index += 1
                    else:
                        current_index += 1
                else:

                    left, right = current_block_start, current_block_end
                    while left <= right:
                        mid = (left + right) // 2
                        if self.tokens[mid] == text:
                            return self._format_results(text)
                        elif self.tokens[mid] < text:
                            left = mid + 1
                        else:
                            right = mid - 1
                    return []
            if level1_positions:
                last_block_end = skip_list[level1_positions[-1]]['end']
                if last_block_end < len(self.tokens) - 1:
                    for i in range(last_block_end + 1, len(self.tokens)):
                        if self.tokens[i] == text:
                            return self._format_results(text)
                        elif self.tokens[i] > text:
                            return []

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

                for i in range(1, len(valid_texts)):
                    next_token = valid_texts[i]
                    next_token_positions = term_results[next_token][file_name]
                    expected_pos = current_pos + 1

                    left, right = 0, len(next_token_positions) - 1
                    found = False
                    while left <= right:
                        mid = (left + right) // 2
                        if next_token_positions[mid] == expected_pos:
                            found = True
                            break
                        elif next_token_positions[mid] < expected_pos:
                            left = mid + 1
                        else:
                            right = mid - 1
                    if not found:
                        is_phrase = False
                        break
                    current_pos = expected_pos
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
        num_tokens = min(num_tokens, len(self.tokens))
        texts = random.sample(self.tokens, num_tokens)
        print(f'随机选择的{num_tokens}个查询token：')
        for i, text in enumerate(texts, 1):
            if i<5:
                print(f'{i}. {text}')
            elif i==num_tokens:
                print(f'...等{i}个token')
        print('\n单层指针测试：')
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
                    print(f' ... 等{len(result) - 2}个文件')
            else:
                print(f'{i}. 查询 "{text}" 未找到结果')

        print('\n' + '-' * 50 + '\n')
        print('双层指针测试：')
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


#
# if __name__ == '__main__':
#     # 指针大小示例
#     skip = skip_pointer(optimizer.load_separate_structures())
#     # print(skip.select_texts('live music', is_one_skip=True))
#     one_skip = skip.create_one_skip_pointer()
#     two_skip = skip.create_two_skip_pointer()
#     print('单层指针', get_dict_size(one_skip), '字节')
#     print('双层指针', get_dict_size(two_skip), '字节')
#     # 运行单次随机查询测试
#     print("跳表索引查询性能测试")
#     print("=" * 50)
#     skip.input_participle(num_tokens=300)
if __name__ == "__main__":
    optimizer = optimize_inverted()

    print("正在生成倒排索引...")
    start_time1 = time.time()
    inverted_index = optimizer.inverted_list()
    print(f"倒排索引生成完成，包含 {len(inverted_index)} 个词项")
    print("\n正在保存分离的词典和倒排表...")
    optimizer.save_separate_structures(inverted_index)
    print("\n正在从分离文件加载倒排索引...")
    start_time2 = time.time()
    loaded_index = optimizer.load_separate_structures()
    end_time = time.time()
    print(f"{end_time-start_time2}s,{start_time2-start_time1}s")
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

    # alll=optimizer.front_coding(tokens=['app', 'appair', 'appal', 'appalachia', 'appalachian', 'appalachians', 'appall','appointment'])
    # print(alll)