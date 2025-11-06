import sys
import time
from init import *

# 查询条件设置
optimizer = optimize_inverted()


class Search:
    def __init__(self):
        pass
        # self.select = init.skip_pointer(inverted_list=inverted_list)
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

        # 处理剩余元素
        while i < len(result1):
            result.append(result1[i]['file_name'])
            i += 1

        while j < len(result2):
            result.append(result2[j]['file_name'])
            j += 1

        return result

    def not_search(self, result1, result2):
        """NOT操作：在result1中但不在result2中的文档"""
        result = []
        i = j = 0

        while i < len(result1):
            file1 = result1[i]['file_name']

            # 在result2中查找file1
            while j < len(result2) and result2[j]['file_name'] < file1:
                j += 1

            # 如果找到匹配项，跳过；否则添加到结果
            if j < len(result2) and result2[j]['file_name'] == file1:
                i += 1
                j += 1
            else:
                result.append(file1)
                i += 1

        return result


def complex_search(condition,is_compress_invarted=False):
    """执行复杂布尔查询"""

    search_instance = Search()
    if is_compress_invarted:
        print("正常倒排表：")
        select_event=skip_pointer(inverted_list=optimizer.inverted_list())
    else:
        print("压缩后的倒排表：")
        select_event=skip_pointer(inverted_list=optimizer.load_separate_structures())
    # 操作符优先级映射
    start_time = time.time()

    precedence = {'+': 1, '-': 1, '*': 2, '(': 0, ')': 0}
    tokens = []
    current_token = ''
    for char in condition:
        if char in ['+', '-', '*', '(', ')']:
            if current_token:
                tokens.append(current_token.strip())
                current_token = ''
            tokens.append(char)
        else:
            current_token += char
    if current_token:
        tokens.append(current_token.strip())

    tokens = [token for token in tokens if token and token != ' ']
    output = []
    operator_stack = []

    for token in tokens:
        if token == '(':
            operator_stack.append(token)
        elif token == ')':
            while operator_stack and operator_stack[-1] != '(':
                output.append(operator_stack.pop())
            if operator_stack and operator_stack[-1] == '(':
                operator_stack.pop()
        elif token in ['+', '-', '*']:
            while (operator_stack and operator_stack[-1] != '(' and
                   precedence[operator_stack[-1]] >= precedence[token]):
                output.append(operator_stack.pop())
            operator_stack.append(token)
        else:  #
            output.append(token)

    while operator_stack:
        output.append(operator_stack.pop())
    term_results = {}
    for term in output:
        if term not in ['+', '-', '*'] and term not in term_results:
            result = select_event.select_text(text=term, is_one_skip=False)
            print(f"{term}: {[item['file_name'] for item in result]}")
            term_results[term] = [item['file_name'] for item in result] if result else []

    result_stack = []

    for item in output:
        if item in ['+', '-', '*']:
            if len(result_stack) < 2:
                print(f"错误: 表达式格式不正确 - 操作符 '{item}' 缺少操作数")
                return []

            operand2 = result_stack.pop()
            operand1 = result_stack.pop()
            # 获取实际的查询结果
            if isinstance(operand1, str):
                result1 = term_results.get(operand1, [])
            else:
                result1 = operand1

            if isinstance(operand2, str):
                result2 = term_results.get(operand2, [])
            else:
                result2 = operand2

            # 执行对应的布尔操作
            if item == '*':
                result = search_instance.and_search(
                    [{'file_name': name} for name in result1],
                    [{'file_name': name} for name in result2]
                )
            elif item == '+':  # OR操作
                result = search_instance.or_search(
                    [{'file_name': name} for name in result1],
                    [{'file_name': name} for name in result2]
                )
            elif item == '-':
                result = search_instance.not_search(
                    [{'file_name': name} for name in result1],
                    [{'file_name': name} for name in result2]
                )
            result_stack.append(result)
        else:
            result_stack.append(item)

    query_time = time.time() - start_time
    if result_stack:
        final_result = result_stack[0] if isinstance(result_stack[0], list) else term_results.get(result_stack[0], [])

        print(f"复杂查询 '{condition}' 的解析结果: {output}")
        print(f"复杂查询 '{condition}' 的结果: {final_result}")
        print(f"查询耗时: {query_time:.4f} 秒")
        return final_result
    else:
        print(f"错误: 无法执行查询 '{condition}'")
        print(f"查询耗时: {query_time:.4f} 秒")
        return []

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




if __name__ == '__main__':
    # complex_search('apple + banana -replace')
    # complex_search('apple + banana * completed',is_compress_invarted=True)
    skip=skip_pointer(optimizer.load_separate_structures())
    print(skip.select_texts('live music',is_one_skip=True))
    one_skip=skip.create_one_skip_pointer()
    two_skip=skip.create_two_skip_pointer()
    print(get_dict_size(one_skip),'字节')
    print(get_dict_size(two_skip),'字节')
