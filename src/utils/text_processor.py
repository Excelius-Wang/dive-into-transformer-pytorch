from pathlib import *
from src.utils.logger import printlog


class TextProcessor:
    def __init__(self,
                 raw_path: str):
        """初始化处理器
        Args:
            raw_path: 原始文本路径
        """
        self.raw_path = raw_path
        # 停用词词表
        self.stopwords_path = str(Path.cwd().parent.parent / 'data/raw/hit_stopwords.txt')
        # 字符到索引的映射
        self.word_to_idx = dict()
        # 索引到字符的映射
        self.idx_to_word = dict()
    def read_text(self):
        try:
            with open(self.raw_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            printlog(f"文件不存在，请检查路径是否正确：{self.raw_path}")
        except Exception as e:
            printlog(f"读取文件时发生错误：{e}")
        return ""

    def get_vocab(self):
        try:
            with open(self.raw_path, 'r', encoding='utf-8') as f:
                # 读取文本内容
                raw_text = f.read().splitlines()
                words = list()
                # 对于每段话，进行词的切分
                for text in raw_text:
                    # 去掉 text 前后的空格
                    text = text.strip()
                    # 去掉 \u3000
                    text = text.replace("\u3000", "").replace(" ", "")
                    # text = self.filter_stopwords(text)
                    # 使用 jieba 进行分词
                    # temp_words = jieba.cut(text, cut_all=False)
                    temp_words = list(set(words))
                    words.extend(temp_words)
                words = sorted(list(set(words)))
                self.word_to_idx = {ch: i for i, ch in enumerate(words)}
                self.idx_to_word = {i: ch for i, ch in enumerate(words)}
                encoding_list = self.encode_text("此开卷第一回也。")
                print(encoding_list)

        except FileNotFoundError:
            printlog(f"文件不存在，请检查路径是否正确：{self.raw_path}")
        except Exception as e:
            printlog(f"读取文件时发生错误：{e}")
        return ""

    def filter_stopwords(self, text):
        with open(self.stopwords_path, 'r', encoding='utf-8') as f:
            stopwords = f.read().splitlines()
            words = list(text)
            filtered_words = [word for word in words if word not in stopwords]
            return ''.join(filtered_words)

    def encode_text(self, text):
        return [self.word_to_idx[word] for word in text]

    def decode_text(self, text):
        return [self.idx_to_word[idx] for idx in text]

hlm_path = Path(Path.cwd().parent.parent, 'data/raw', 'HLM_Short.txt')
printlog("红楼梦路径为：" + str(hlm_path))
text_processor = TextProcessor(str(hlm_path))
text_processor.get_vocab()
# print(text_processor.read_text())
