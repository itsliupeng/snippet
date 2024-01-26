import os
import jieba


def convert_to_fasttext_format(root_dir, output_file):
    """
    Convert a directory of text files into FastText format, tokenizing Chinese text using jieba.

    Parameters:
    - root_dir (str): Root directory containing subdirectories named as labels.
                      Each subdirectory contains text files.
    - output_file (str): File path to save the converted dataset.
    """

    def tokenize_chinese_text(text):
        """Tokenize Chinese text using jieba."""
        return " ".join(jieba.cut(text))

    with open(output_file, 'w', encoding='utf-8') as out_f:
        # Iterate over all directories (labels)
        for label in os.listdir(root_dir):
            label_dir = os.path.join(root_dir, label)
            if os.path.isdir(label_dir):
                # Iterate over all files in the directory
                for filename in os.listdir(label_dir):
                    file_path = os.path.join(label_dir, filename)
                    with open(file_path, 'r', encoding='utf-8') as in_f:
                        content = in_f.read().strip().replace('\n', '\\n')  # Escape newlines
                        tokenized_content = tokenize_chinese_text(content)
                        out_f.write(f"__label__{label} {tokenized_content}\n")

# Usage:
# convert_to_fasttext_format('path_to_your_root_directory', 'output_fasttext_format.txt')
