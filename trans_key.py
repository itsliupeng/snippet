
in_file = 'article.jsonl'
out_file = 'article_content.jsonl'
out_f = open(out_file, 'w')
for line in open(in_file):
    line = line.replace('"text"', '"content"')
    out_f.write(line)
out_f.close()