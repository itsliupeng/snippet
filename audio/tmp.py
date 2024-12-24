import os
import sys


def gen_list(root_dirs, done_dirs, todo_file, done_file):
    todo, done = 0, 0
    with open(todo_file, 'w') as o1:
        with open(done_file, 'w') as o2:
            for root_dir in root_dirs:
                for dirpath, dirnames, filenames in os.walk(root_dir):
                    if not dirnames:  # no more subdir
                        for fname in filenames:
                            for suffix in ['.m4a', '.mp3']:
                                if not fname.endswith(suffix): continue
                                outname = None
                                for done_dir in done_dirs:
                                    tmp = os.path.join(dirpath, fname[:-len(suffix)], 'transcription.jsonl')
                                    output = tmp.replace(root_dir, done_dir)
                                    if os.path.exists(output):
                                        outname = output
                                        break
                                if outname:
                                    #print(f'Existing {outname=}')
                                    done += 1
                                    o2.write(os.path.join(dirpath, fname) + ' -> ' + outname + '\n')
                                else:
                                    todo += 1
                                    o1.write(os.path.join(dirpath, fname) + '\n')

    print(f'Wrote {todo_file}: {todo=}')
    print(f'Wrote {done_file}: {done=}')


if __name__ == '__main__':
    gen_list(
        ['/gpfs/public/pretrain/data/audio/spotify/stat_date=20241203/',
         '/gpfs/public/pretrain/data/audio/spotify/stat_date=20241204/',
         "/gpfs/public/mmodal/users/liupeng/data/spotify/stat_date=20241205/",
         "/gpfs/public/mmodal/users/liupeng/data/spotify/stat_date=20241206/",
         "/gpfs/public/mmodal/users/liupeng/data/spotify/stat_date=20241207/",
         "/gpfs/public/mmodal/users/liupeng/data/spotify/stat_date=20241208/",
         "/gpfs/public/mmodal/users/liupeng/data/spotify/stat_date=20241212/",
         "/gpfs/public/mmodal/users/liupeng/data/spotify/stat_date=20241213/",
         "/gpfs/public/mmodal/users/liupeng/data/spotify/stat_date=20241216/",
         "/gpfs/public/mmodal/users/liupeng/data/spotify/stat_date=20241217/",
         "/gpfs/public/mmodal/users/liupeng/data/spotify/stat_date=20241218/",
         "/gpfs/public/mmodal/users/liupeng/data/spotify/stat_date=20241219/"
         ],
        ['/gpfs/public/pretrain/data/audio/spotify_processed/20241203_20241219/'],  # trailing '/' is necessary
        'audio.todo.txt', 'audio.done.txt')