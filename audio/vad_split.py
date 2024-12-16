import argparse
import json
import multiprocessing
import os
from pydub import AudioSegment
from pydub.utils import mediainfo
import whisper
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps


silero_model = load_silero_vad()
def silero_vad(file_path):
    wav = read_audio(file_path)
    speech_timestamps = get_speech_timestamps(
      wav,
      silero_model,
      return_seconds=True,
    )
    speech_ranges = [[t['start'], t['end']] for t in speech_timestamps]
    return speech_ranges


def get_kbps(inaudio):
    try:
        info = mediainfo(inaudio)
        kbps = f"{int(int(info['bit_rate']) / 1000)}k"  # Convert to kbps string format (e.g., "57k")
        print(f'{inaudio}: {kbps=}')
        return kbps
    except KeyError:
        return '64k'


def split_audio(inaudio, outdir, time_segments, kbps):
    os.makedirs(outdir, exist_ok=True)
    ext = os.path.splitext(inaudio)[1]
    assert len(ext) > 1
    ext = ext[1:]
    audio = AudioSegment.from_file(inaudio, format=ext)
    outfiles = []
    for i, (start_sec, end_sec) in enumerate(time_segments):
        start_ms = start_sec * 1000
        end_ms = end_sec * 1000
        clip = audio[start_ms:end_ms]

        #outwav = os.path.join(outdir, f"clip_{i+1}.wav")
        #clip.export(outwav, format="wav")
        #print(f"Exported: {outwav}")
        #outfiles.append(outwav)

        outfile = os.path.join(outdir, f"clip_{i+1}.mp3")
        clip.export(outfile, format="mp3", bitrate=kbps)
        outfiles.append(outfile)
    return outfiles


whisper_model = whisper.load_model("/gpfs/public/01/models/audio/whisper-large-v3/large-v3-turbo.pt")
def whisper_transcribe(audio_files):
    data = []
    for fname in audio_files:
        # options = {
        #     'task':'transcribe',
        #     'language':'en',
        #     'temperature':0.0,
        #     #'best_of':5,
        #     'beam_size':5,
        # }
        # result = whisper_model.transcribe(fname, verbose=False, **options)
        result = whisper_model.transcribe(fname)
        d = {'wav':fname, 'whisper':result['text']}
        data.append(d)
    return data


def segment_time_window(vad_list, winsize):
    time_segments = []
    seg = None
    for vad in vad_list:
        if seg is None:
            seg = vad
            continue
        if vad[1] - seg[0] <= winsize:
            seg[1] = vad[1]
        else:
            time_segments.append(seg)
            seg = vad
    if seg: time_segments.append(seg)
    #print(f'{vad_list=}')
    #print(f'{time_segments=}')
    return time_segments


def process_one_file(inaudio, workdir, winsize=30):

    # prefix = '/gpfs/public/pretrain/data/audio/VoiceAssistant-400K/audio/'
    # suffix = '.wav'
    # assert inaudio.startswith(prefix) and inaudio.endswith(suffix), print(inaudio)
    # outdir = os.path.join(workdir, inaudio[len(prefix):-len(suffix)])
    bname = os.path.splitext(os.path.basename(inaudio))[0]
    outdir = os.path.join(workdir, bname)

    try:
        vad_list = silero_vad(inaudio)
    except Exception as e:
        print(f'{inaudio}: silero_vad error {e}')
        return
    kbps = get_kbps(inaudio)
    time_segments = segment_time_window(vad_list, winsize=winsize)
    try:
        audio_files = split_audio(inaudio, outdir, time_segments, kbps)
    except Exception as e:
        print(f'{inaudio}: split_audio error {e}')
        return
    try:
        trans = whisper_transcribe(audio_files)
    except Exception as e:
        print(f'{inaudio}: whisper_transcribe error {e}')
        return
    outjson = os.path.join(outdir, 'transcription.jsonl')
    with open(outjson, 'w') as o:
        for d,t in zip(trans, time_segments):
            d['audio'] = inaudio
            d['time'] = t
            d['duration'] = t[1] - t[0]
            o.write(json.dumps(d, ensure_ascii=False) + '\n')


def process_wrapper(args):
    injson, workdir = args
    print(f'Processing {injson}')
    process_one_file(injson, workdir)


def process_mp(infile_list, outdir, winsize):
    with multiprocessing.Pool(2) as pool:
        pool.map(process_wrapper, [(infile, outdir, winsize) for infile in infile_list])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inlist', type=str, default='')
    parser.add_argument('--begin', type=int, default=-1)
    parser.add_argument('--end', type=int, default=-1)
    parser.add_argument('--outdir', type=str, default='')
    parser.add_argument('--winsize', type=int, default=30)
    args = parser.parse_args()

    with open(args.inlist) as f:
        infiles = [l.strip() for l in f.readlines()]
    for i,l in enumerate(infiles):
        if args.begin <= i < args.end:
            infile = l
            print(f"Processing file {i}: {infile}")
            process_one_file(infile, args.outdir, args.winsize)

    # infile_list = [l.strip() for l in lines[args.begin:args.end]]
    # process_mp(infile_list, args.outdir, args.winsize)