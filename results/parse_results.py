import os
dir_list = os.listdir()

def remove_extra_info(txt_file):
    with open(txt_file, 'r') as f_read, open(txt_file[:-4] + '_lean.txt', 'w+') as f_write:
        for line in f_read:
            if 'list' in line:
                break
            f_write.write(line)

def delete_original_file(txt_file):
    if 'lean' not in txt_file:
        os.remove(txt_file)


def find_files_recursive(path, folder):
    if folder is '':
        folder_path = path
    else:
        folder_path = path + '/' + folder
    print(folder_path)
    print(os.listdir(folder_path))
    for thing in os.listdir(folder_path):
        print(folder_path + '/' + thing)
        print(os.path.isdir(folder_path + '/' + thing))
        if os.path.isdir(folder_path + '/' + thing):
            find_files_recursive(folder_path, thing)
        elif thing[-3:] == 'txt':
            #remove_extra_info(folder_path + '/' + thing)
            delete_original_file(folder_path + '/' + thing)


def main():
    find_files_recursive('./', '')

if __name__ == '__main__':
    main()
