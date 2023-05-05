import os
import random
import tqdm
import argparse


def findAllSeqs_all_clean(dirName,
                save_list_path='data',
                extension='.wav',
                speaker_level=1,
                enroll_spks=100,
                enroll_utts=4):

    if dirName[-1] != os.sep:
        dirName += os.sep
    prefixSize = len(dirName)
    speakersTarget = {}

    print("dir_name",dirName) #/mnt/a3/cache/database/voxceleb/vox2/dev/acc/
    print(f"finding {extension}, Waiting...")
    f_e = open(f"{save_list_path}/list3/enroll_spks{enroll_spks}_utts{enroll_utts}.lst",'w')
    f_t = open(f"{save_list_path}/list3/test_spks{enroll_spks}_utts{enroll_utts}_clean.lst",'w')
    f_t_canteen = open(f"{save_list_path}/list3/test_spks{enroll_spks}_utts{enroll_utts}_canteen.lst", 'w')
    f_t_road = open(f"{save_list_path}/list3/test_spks{enroll_spks}_utts{enroll_utts}_road.lst", 'w')

    for root, dirs, filenames in tqdm.tqdm(os.walk(dirName, followlinks=True)):
        # print("root, dirs, filenames:",root, dirs, filenames)

        filtered_files = [f for f in filenames if f.endswith(extension)]
        if len(filtered_files) > 0:

            speakerStr = (os.sep).join(
                root[prefixSize:].split(os.sep)[:speaker_level]) ##控制spk_id:id03439,id03439/tybcebyc

            audio_type = speakerStr.split('/')[1]
            spkid = speakerStr.split('/')[0]
            speakerStr_haisi = spkid
            if audio_type== "clean":
                # print("str:", speakerStr)
                if speakerStr_haisi not in speakersTarget:
                    speakersTarget[speakerStr_haisi] = len(speakersTarget)
                    # speaker2utt[speakerStr_haisi] = []
                    if len(speakersTarget)>enroll_spks:
                        break

                    enroll_list = filtered_files[:enroll_utts]
                    test_list = filtered_files[enroll_utts:]

                    f_e.write(speakerStr_haisi)
                    for utt1 in enroll_list:
                        full_path = os.path.join(speakerStr, utt1)
                        f_e.write('\t')
                        f_e.write(full_path)
                    f_e.write('\n')

                else:
                    test_list = filtered_files

                for utt2 in test_list:
                    full_path = os.path.join(speakerStr, utt2)
                    f_t.write(full_path)
                    f_t.write('\n')

            elif audio_type== "canteen":
                test_list = filtered_files
                for utt2 in test_list:
                    full_path = os.path.join(speakerStr, utt2)
                    f_t_canteen.write(full_path)
                    f_t_canteen.write('\n')

            elif audio_type== "road":
                test_list = filtered_files
                for utt2 in test_list:
                    full_path = os.path.join(speakerStr, utt2)
                    f_t_road.write(full_path)
                    f_t_road.write('\n')

            # else:
            #     test_list = filtered_files
            #     for utt2 in test_list:
            #         full_path = os.path.join(speakerStr, utt2)
            #         f_t_canteen.write(full_path)
            #         f_t_canteen.write('\n')

    f_e.close()
    f_t.close()
    f_t_canteen.close()
    f_t_road.close()


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--testset_root', help='测试数据路径', type=str)
    parser.add_argument('--save_list_path', help='保存文件路径', type=str, default="data")
    parser.add_argument('--extension', help='file extension name', type=str, default="wav")
    parser.add_argument('--speaker_level', help='说话人id', type=int, default=1)
    parser.add_argument('--enroll_spks', help='注册说话人数量', type=int, default=100)
    parser.add_argument('--enroll_utts', help='注册语音数量', type=int, default=5)

    args = parser.parse_args()

    # args.testset_root = r"E:\Dataset\数据查看\海思\数据\test-20-people\test-20-people"
    # args.save_list_path = r"E:\Dataset\数据查看\海思\数据\test-20-people"

    if not os.path.exists(f"{args.save_list_path}/list3"):
        os.mkdir(f"{args.save_list_path}/list3")

    findAllSeqs_all_clean(args.testset_root,
                save_list_path=args.save_list_path,
                extension=args.extension,
                speaker_level=args.speaker_level,
                enroll_spks=args.enroll_spks,
                enroll_utts=args.enroll_utts)

