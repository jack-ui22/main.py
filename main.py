from search import *
from init import create_token, optimize_inverted
from skip_pointer import skip_pointer
#optimizer=optimize_inverted()


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

def main():
    interactive_search()

if __name__ == "__main__":
    try:
        import nltk
        nltk.data.find('tokenizers/punkt_tab')
    except (ImportError, LookupError):
        print("正在下载必要的NLTK数据...")
        import nltk
        nltk.download('punkt_tab')
    main()



