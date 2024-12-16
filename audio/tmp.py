import json
import glob
from langdetect import detect
from multiprocessing import Pool, cpu_count


def process_line(line):
    line = line.strip()
    if not line:
        return None
    try:
        data = json.loads(line)
        # Perform language detection on the 'refined' field
        if 'refined' in data:
            data['lang'] = detect(data['refined'])
        return data
    except:
        return None


def process_file(input_file):
    # Create output file name by appending '_lang' before the extension
    output_file = "part_filter/" + input_file.split("/")[-1] + '_lang.jsonl'

    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            result = process_line(line)
            if result is not None:
                outfile.write(json.dumps(result, ensure_ascii=False) + '\n')

    return output_file


if __name__ == '__main__':
    # Get the list of all jsonl files in part_dir
    input_files = glob.glob('part/part_*')

    # Adjust the number of processes based on your system's cores
    num_processes = min(cpu_count(), len(input_files))

    with Pool(processes=num_processes) as pool:
        results = pool.map(process_file, input_files)

    print("Processing complete. Created these files:")
    for r in results:
        print(r)
