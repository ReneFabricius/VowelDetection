import os
import os.path
import csv
import numpy as np
from PIL import Image
from timeit import default_timer as timer
import librosa
import soundfile as sf
import sys
import cupy as cp
from scipy import signal as scisig


def generate_spectrograms(
        NSAMP=1000,                     # Number of created spectrograms per class (vowel, non-vowel)
        N=512,                          # Window width in samples
        fbefore=250,                    # Number of windows before the center window
        fafter=250,                     # Number of windows after the center window
        ds=2,                           # Step between the windows in samples
        windows=None,                   # Matrix of window functions, 3 rows, N columns.
        dynRange=60,                    # Dynamic range
        signalLevel=10,                 # SNR
        sourceDir="",                   # TIMIT dataset audio folder
        sourceDirNoise="",              # Noise dataset audio folder
        targetDir="",                   # Output directory
        noiseType="MAVD",               # Type of the noise used {"MAVD", "ESC50"}
        batchSize=50,                   # Batch size when computing fft and creating spectrograms
        debug=False

):
    start_time = timer()
    if windows is None:
        windows = cp.ones(shape=(3, N))
    else:
        windows = cp.array(windows)

    eps = 1e-10
    phones = []
    wavfiles = []

    minstart = N // 2 + fbefore * ds

    for path, subdirs, files in os.walk(sourceDir):
        for name in files:
            basename, ext = os.path.splitext(name)
            fname = os.path.join(path, name)
            if ext != ".PHN":
                continue
            with open(fname, newline='') as phnfile:
                rdr = list(csv.reader(phnfile, delimiter=' '))
                wname = os.path.join(path, "{0}.WAV".format(basename))
                file_phones = []
                last_sampl = int(rdr[-1][1])
                maxstart = last_sampl - ds * fafter - N // 2 + 1
                for row in rdr:
                    start = int(row[0])
                    end = int(row[1])
                    if start < minstart:
                        continue
                    if end > maxstart:
                        continue
                    data = (wname, start, end, row[2].strip())
                    file_phones.append(data)

                phones += file_phones
                wavfiles.append(wname)

    print("Total {0} phones in {1} files".format(len(phones), len(wavfiles)))

    vowels = ["iy", "ih", "eh", "ey", "ae", "aa", "aw", "ay", "ah",
              "ao", "oy", "ow", "uh", "uw", "ux", "er", "ax", "ix", "axr",
              "ax-h"]

    # do not take samples from the very edges
    margin = int((fbefore + fafter) * ds * 0.1)
    phn_len = lambda x: max(x[2] - x[1] - margin, 1)

    class PhonePos:
        def __init__(self, phone, kind):
            self.file = phone[0]
            self.start = phone[1]
            self.end = phone[2]
            self.phone = phone[3]
            self.kind = kind

    def random_pos(kind="vowel"):
        if kind == "vowel":
            subset_iter = (filter(lambda x: x[3] in vowels, phones))
        else:
            subset_iter = (filter(lambda x: x[3] not in vowels, phones))
        subset_phones = list(subset_iter)
        subset_cs = np.cumsum(np.array(list(map(phn_len, subset_phones))))
        subset_max = subset_cs[-1]
        print(kind + " phonemes combined length: " + str(subset_max))

        subset_pos0 = np.sort(np.random.choice(subset_max, size=NSAMP, replace=False))
        subset_idx = np.searchsorted(subset_cs, subset_pos0, side='right')

        subset_pos = []
        for i in range(len(subset_idx)):
            j = subset_idx[i]
            assert (j >= 0)
            assert (j < len(subset_cs))
            phone = subset_phones[j]
            pp = PhonePos(phone, kind)

            # position within file
            file_pos = subset_pos0[i] + phone[1] + margin // 2

            if j > 0:
                file_pos -= subset_cs[j - 1]

            assert (file_pos <= phone[2] - margin // 2)
            assert (file_pos >= phone[1] + margin // 2)
            pp.file_pos = file_pos

            subset_pos.append(pp)

        # returns a list of PhonePos objects
        return subset_pos

    def list_noise_files(sound_ext=".wav"):
        noise_path_list = []
        for path, subdirs, files in os.walk(sourceDirNoise):
            for name in files:
                # print("Processing folder: {}".format(path))
                basename, ext = os.path.splitext(name)
                if ext != sound_ext:
                    continue
                fname = os.path.join(path, name)
                noise_path_list.append(fname)

        print(str(len(noise_path_list)) + " noise files found")
        return noise_path_list

    def load_noises(ns_pl, tar_rate):
        print("Loading noise")
        noises = []
        for ni, nois_p in enumerate(ns_pl):
            print("Loading noise " + str(int(ni / len(ns_pl) * 100)) + "%", end="\r")
            nois, rate = librosa.load(nois_p, sr=None, mono=True)
            nois_rs = librosa.resample(nois, rate, tar_rate)
            noises.append(nois_rs)

        return np.concatenate(noises)

    def tile_noise(noise, tar_len):
        multiple = tar_len / len(noise)
        repeat_c = int(np.floor(multiple))
        rest = tar_len - repeat_c * len(noise)
        return np.concatenate([np.repeat(noise, repeat_c), noise[:rest]], 0)

    def combine_with_noise(sig, nois, snr):
        # snr = 10*log10(sum(s**2)/sum(n**2))
        if len(nois) != len(sig):
            nois = tile_noise(nois, len(sig))

        E_sig = sum(sig ** 2)
        E_nois = sum(nois ** 2)
        if E_nois == 0:
            print("Warning: zero energy noise")
            return sig

        if type(snr) == list:
            snr = np.random.uniform(snr[0], snr[1])

        coef = 10 ** ((snr - 10 * np.log10(E_sig / E_nois)) / (-20))
        return (sig + coef * nois) / (1 + coef)

    vp = random_pos("vowel")
    up = random_pos("nonvowel")

    v_tarDir = os.path.join(targetDir, "vowel")
    u_tarDir = os.path.join(targetDir, "nonvowel")

    if not os.path.exists(v_tarDir):
        os.makedirs(v_tarDir)
    if not os.path.exists(u_tarDir):
        os.makedirs(u_tarDir)

    # timit sample rate is 16 kHz
    t_rate = 16000

    if signalLevel is not None:
        if noiseType == "MAVD":
            noise_files = list_noise_files(".flac")
            noises = load_noises(noise_files, t_rate)
        elif noiseType == "ESC50":
            nf = list_noise_files()

    complete_count = 0

    for wavFile in wavfiles:
        curList = []
        while len(vp) > 0 and vp[0].file == wavFile:
            pp = vp.pop(0)
            curList.append(pp)

        while len(up) > 0 and up[0].file == wavFile:
            pp = up.pop(0)
            curList.append(pp)

        if len(curList) == 0:
            continue

        complete_count += len(curList)
        print("Creating spectrograms " + str(int(complete_count / (2 * NSAMP) * 100)) + "%", end="\r")

        with open(wavFile, "rb") as fp:
            fp.read(1024)  # need to jump over 1024 bytes
            wa = fp.read()
            snd = np.frombuffer(wa, dtype=np.int16)

        snd = snd.astype(np.float32) / 2 ** 15

        if signalLevel is not None:
            if noiseType == "MAVD":
                noise_start = np.random.randint(0, len(noises) - len(snd))
                nois = noises[noise_start:noise_start + len(snd)]

            elif noiseType == "ESC50":
                noise_ind = np.random.randint(len(nf))
                nois, nois_rate = librosa.load(nf[noise_ind], sr=None, mono=False)
                nois = librosa.resample(nois, nois_rate, t_rate, res_type='kaiser_fast')


            snd = combine_with_noise(snd, nois, signalLevel)

        ftotal = 1 + fbefore + fafter
        dat = np.zeros(shape=(len(curList), 3, ftotal, N), dtype=cp.float32)

        if debug:
            sound_fn = os.path.join(targetDir, reg + "_" + speaker + "_" + fname + ".wav")
            sf.write(sound_fn, snd, samplerate=t_rate)

        for i, pp in enumerate(curList):
            p = pp.file_pos
            start = p - ds * fbefore - N // 2
            end = p + ds * fafter + N // 2
            stride_bytes = snd.strides[0]
            matrix = np.lib.stride_tricks.as_strided(snd[start:end],
                                                     shape=(ftotal, N),
                                                     strides=(stride_bytes * ds, stride_bytes),
                                                     writeable=False)
            dat[i, :, :, :] = matrix

        for bstart in range(0, dat.shape[0], batchSize):
            dat_batch = cp.transpose(cp.array(dat[bstart:bstart + batchSize]), (0, 2, 1, 3))
            spect_abs = cp.abs(cp.fft.rfft(cp.multiply(dat_batch, windows), axis=3))
            spect_abs[spect_abs == 0] = eps
            spectra = 20 * cp.log10(spect_abs)
            maxs = cp.max(spectra, axis=(1, 2, 3), keepdims=True)
            mins = cp.max(cp.concatenate((maxs - dynRange, cp.min(spectra, axis=(1, 2, 3), keepdims=True)), 3), 3,
                          keepdims=True)
            M_sp = spectra > mins
            spectra[~M_sp] = 0
            spectra[M_sp] = (((spectra - mins) / (maxs - mins)) * 255)[M_sp]

            spec_tr = cp.flip(cp.transpose(spectra, (0, 3, 1, 2)), 1).astype(dtype=cp.byte)

            fname = os.path.splitext(os.path.basename(wavFile))[0]
            reg, speaker = os.path.split(os.path.split(wavFile)[0])
            reg = os.path.split(reg)[1]

            for i, pp in enumerate(curList[bstart:bstart + batchSize]):
                pos = pp.file_pos
                img_name = reg + "_" + speaker + "_" + fname + "_" + str(pos) + "_" + str(pp.phone) + ".png"
                if pp.kind == "vowel":
                    img_path = os.path.join(v_tarDir, img_name)
                else:
                    img_path = os.path.join(u_tarDir, img_name)

                img = Image.fromarray(spec_tr[i].get(), mode="RGB")
                img.save(img_path)

    print("Finished in: {}s".format(timer() - start_time))


if __name__ == "__main__":
    if len(sys.argv) != 7:
        print("Parameters are: signal_audio_folder noise_audio_folder output_folder noise_type samples_per_cl SND" )
        exit(1)

    sig_fold = sys.argv[1]
    nois_fold = sys.argv[2]
    out_fold = sys.argv[3]
    nois_type = sys.argv[4]

    if nois_type not in {"MAVD", "ESC50"}:
        print("Possible values for noise_type parameter are: {\"MAVD\", \"ESC50\"}")
        exit(1)

    n_samp = int(sys.argv[5])
    SND = sys.argv[6]
    try:
        if SND == "None":
            SND = None
        elif SND[0] == "[" and SND[-1] == "]":
            SND = [float(a) for a in SND.strip("][").split(",")]
            if len(SND) != 2:
                raise ValueError("Error: wrong number os elements in SND")
        else:
            SND = float(SND)
    except ValueError:
        print("Possible values for SND are None, float number or a list of two float numbers")
        exit(1)

    with cp.cuda.Device(0):
        N = 512
        W = np.zeros((3, N))
        W[0] = scisig.nuttall(N, sym=False)
        W[1] = scisig.gaussian(N, N / 8, sym=False)
        W[2] = scisig.blackmanharris(N, sym=False)
        generate_spectrograms(N=N, windows=W, targetDir=out_fold, sourceDir=sig_fold,
                     sourceDirNoise=nois_fold, NSAMP=n_samp, debug=False, signalLevel=SND,
                              noiseType=nois_type)

