import time
import math
from init import *
from skip_pointer import skip_pointer

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

            while j < len(result2) and result2[j]['file_name'] < file1:
                j += 1

            if j < len(result2) and result2[j]['file_name'] == file1:
                i += 1
                j += 1
            else:
                result.append(file1)
                i += 1

        return result


def complex_search(condition,is_compress_invarted=True):
    """执行复杂布尔查询"""

    search_instance = Search()
    if not is_compress_invarted:
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
        else:
            output.append(token)

    while operator_stack:
        output.append(operator_stack.pop())
    term_results = {}
    for term in output:
        if term not in ['+', '-', '*'] and term not in term_results:
            result = select_event.select_text(text=term, is_one_skip=False)
            print(f"{term}: {len(result)},{[item['file_name'] for item in result]}")
            term_results[term] = [item['file_name'] for item in result] if result else []
            if not result:
                print(f'{term} 可能为短语')
                result ,times= select_event.select_texts(texts = term,is_one_skip=True)
                result = [item['file_name'] for item in result]
                print(f'{term}:',len(result))

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
        return final_result,query_time
    else:
        print(f"错误: 无法执行查询 '{condition}'")
        print(f"查询耗时: {query_time:.4f} 秒")
        return [],query_time



class tf_idf:
    def __init__(self, inverted_list=None):
        self.inverted_list = inverted_list if inverted_list is not None else optimizer.load_separate_structures()
        self.total_files = 93513
        print(f"倒排索引加载完成，包含 {len(self.inverted_list)} 个词项")
        self._docs_list = None
        self._idf_values = None
        self._doc_vectors = None
    
    @property
    def docs_list(self):
        """
        延迟加载文档列表
        """
        if self._docs_list is None:
            self._docs_list = self._get_all_documents()
        return self._docs_list
    
    @property
    def idf_values(self):
        """
        延迟加载IDF值
        """
        if self._idf_values is None:
            self._idf_values = self._calculate_idf()
        return self._idf_values
    
    @property
    def doc_vectors(self):
        """
        延迟加载文档向量
        """
        if self._doc_vectors is None:
            self._doc_vectors = self._build_document_vectors()
        return self._doc_vectors
    
    def _get_all_documents(self):
        """
        获取所有文档的列表
        """
        docs_set = set()
        for token, docs in tqdm(self.inverted_list.items(), desc="收集文档ID", total=len(self.inverted_list)):
            docs_set.update(docs.keys())
        
        docs_list = list(docs_set)
        print(f"找到 {len(docs_list)} 个文档")
        return docs_list
    
    def _calculate_idf(self):
        """
        计算所有词项的IDF值
        """
        print('计算idf')
        idf_values = {}
        total_docs = self.total_files
        for token in tqdm(self.inverted_list, desc="计算IDF值", total=len(self.inverted_list)):
            doc_freq = len(self.inverted_list[token])
            idf = math.log(total_docs / (doc_freq + 1)) + 1  # 加1平滑
            idf_values[token] = idf
        
        print("IDF值计算完成")
        return idf_values
    
    def _build_document_vectors(self):
        """
        构建文档的TF-IDF向
        采用反向遍历策略：对每个词项，只处理包含它的文档
        """
        print(f"开始构建文档TF-IDF向量，总词项数: {len(self.inverted_list)}, 总文档数: {len(self.docs_list)}")

        doc_vectors = {}
        for doc in self.docs_list:
            doc_vectors[doc] = {}

        for token in tqdm(self.inverted_list.items(), desc="处理词项", total=len(self.inverted_list)):
            token_name, token_docs = token[0], token[1]
            idf = self.idf_values.get(token_name, 0)
            for doc in token_docs:
                tf = len(token_docs[doc])
                doc_vectors[doc][token_name] = tf * idf

        for doc in tqdm(doc_vectors.items(), desc="归一化向量", total=len(doc_vectors)):
            doc_name, vector = doc[0], doc[1]
            norm = math.sqrt(sum(val**2 for val in vector.values()))

            if norm > 0:
                for token in vector:
                    vector[token] /= norm
        
        print("文档TF-IDF向量构建完成")
        return doc_vectors
    
    def _calculate_tf_for_doc(self, token, doc):
        """
        计算特定词项在特定文档中的TF值
        """
        if token not in self.inverted_list or doc not in self.inverted_list[token]:
            return 0
        return len(self.inverted_list[token][doc])
    
    def _create_query_vector(self, query_tokens):
        """
        创建查询向量
        """
        query_freq = {}
        for token in query_tokens:
            query_freq[token] = query_freq.get(token, 0) + 1

        query_vector = {}
        for token, freq in query_freq.items():
            tf = freq
            idf = self.idf_values.get(token, 0)
            query_vector[token] = tf * idf
        

        norm = math.sqrt(sum(val**2 for val in query_vector.values()))
        if norm > 0:
            for token in query_vector:
                query_vector[token] /= norm
        
        return query_vector
    
    def _cosine_similarity(self, vec1, vec2):
        """
        计算两个向量的余弦相似度
        """
        dot_product = 0
        for token in vec1:
            if token in vec2:
                dot_product += vec1[token] * vec2[token]

        norm1 = math.sqrt(sum(val**2 for val in vec1.values()))
        norm2 = math.sqrt(sum(val**2 for val in vec2.values()))

        if norm1 == 0 or norm2 == 0:
            return 0
        
        return dot_product / (norm1 * norm2)
    
    def tf(self, texts):
        """
        计算文本中的词项并获取它们在倒排索引中的位置信息
        """
        clean = create_token()
        skip = skip_pointer(self.inverted_list)
        stopwords = clean.read_stopwords()
        tokens = nltk.word_tokenize(texts)
        valid_texts = clean.clean_token(tokens=tokens, stopwords=stopwords)
        print('清理后的结果:', valid_texts)
        
        result = []
        for text in valid_texts:
            positions = skip.select_text(text=text, is_one_skip=True)
            print(positions)
            result.extend(positions)
        
        return result
    
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

if __name__ == '__main__':
    tfidf = tf_idf(inverted_list=optimizer.load_separate_structures())

    print("\nTF-IDF文档检索")
    sample_queries = [
        "live music apple ",
        "banana apple  ",
        'music festival ',
        'amazing '
    ]
    for query in sample_queries:
        print(f"\n执行查询: '{query}'")
        tfidf.search(query, top_k=5)

    # complex_search('music-rain+festival-outdoor+concert')
    # complex_search('(festival-rain+(concert-outdoor))')
    # complex_search('((music+concert)+festival)-(rain+outdoor)')

    # complex_search('(basketball*tournament)+(football*match)-canceled')
    #complex_search('', is_compress_invarted=True)
    # skip=skip_pointer(optimizer.load_separate_structures())
    # name,times=skip.select_texts('outdoor sports',is_one_skip=True)
    # print(name)
    # print(times)

