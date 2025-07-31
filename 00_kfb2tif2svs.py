import os
import sys
import subprocess
from time import time
from glob import glob


def main():
    src_folder_name = "./samples/00_kfb"
    des_folder_name = "./samples/01_svs"

    level = 8

    exe_path = r'W:\kfb2tif2svs\x86\KFbioConverter.exe'
    if not os.path.exists(exe_path):
        raise FileNotFoundError('Could not find convert library.')

    pwd = r'W:\Data\WSL'
    full_path = os.path.join(pwd, src_folder_name)
    dest_path = os.path.join(pwd, des_folder_name)

    if not os.path.exists(full_path):
        raise FileNotFoundError(f'could not get into dir {src_folder_name}')
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)

    # 使用glob获取更可靠的文件列表
    kfb_list = glob(os.path.join(full_path, "*.kfb"))
    kfb_list = [os.path.basename(f) for f in kfb_list]

    print(f'Found {len(kfb_list)} slides, transferring to svs format...')
    for elem in kfb_list:
        st = time()
        kfb_elem_path = os.path.join(full_path, elem)
        svs_dest_path = os.path.join(dest_path, elem.replace('.kfb', '.svs'))

        if os.path.exists(svs_dest_path):
            continue

        # 添加BigTIFF支持参数（根据实际工具支持的参数调整）
        command = [
            exe_path,
            kfb_elem_path,
            svs_dest_path,
            str(level)#,
            # "--bigtiff"  # 添加BigTIFF支持参数
        ]
        command = ' '.join(command)
        print(f'Processing {elem}...')

        # 使用subprocess.run代替Popen获取更好控制
        result = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        print(f'Success: {result.stdout}')

        print(f'Finished {elem}, time: {time() - st:.2f}s')


if __name__ == "__main__":
    main()