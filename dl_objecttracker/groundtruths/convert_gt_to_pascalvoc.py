from os import path

# groundtruth format required: <class_name> <left> <top> <right> <bottom>


# ToDo: modify according to dataset gt format
def get_files(dataset_name, path_file, previous_frame, target):
    if dataset_name == 'mot':  # ToDo: include ids in tracking (pending); now not taking id into account
        folder_path = path_file.rsplit('/', 1)[0]
        with open(path_file, 'r') as stream:
            out = stream.readlines()  # <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <object_type>, <vis_ratio>
            for i in range(len(out)):
                out[i] = out[i].split(',')
                with open(folder_path + '/' + str(int(out[i][0]) - 1) + '.txt', 'a') as logfile:
                    logfile.write(
                        target + " " + str(out[i][2]) + " " + str(out[i][3]) + " " + str(int(
                            out[i][2]) + int(out[i][4])) + " " + str(int(out[i][3]) + int(out[i][5])) + "\n")

    elif dataset_name == 'otb':
        folder_path = path_file.rsplit('/', 1)[0]
        with open(path_file, 'r') as stream:
            out = stream.readlines()  # (x, y, box-width, box-height)
            for i in range(len(out)):
                out[i] = out[i].split(',')
                with open(folder_path + '/' + str(i) + '.txt', 'a') as logfile:
                    logfile.write(
                        target + " " + str(out[i][0]) + " " + str(out[i][1]) + " " + str(int(
                            out[i][0])+int(out[i][2])) + " " + str(int(out[i][1])+int(out[i][3][:2])) + "\n")
    elif dataset_name == 'nfs':
        folder_path = path_file.rsplit('/', 1)[0]
        with open(path_file, 'r') as stream:
            out = stream.readlines()  # (*, left, top, right, bottom, frame, *, *, *, class)
            for i in range(len(out)):
                out[i] = out[i].split(' ')
                with open(folder_path + '/' + str(i) + '.txt', 'w') as logfile:
                    logfile.write(target + " " + str(out[i][1]) + " " + str(out[i][2]) + " " + str(int(out[i][3])) + " " + str(int(out[i][4])) + "\n")


if __name__ == "__main__":
    file_path = None
    target = None
    dataset = 'mot'
    if dataset == 'mot':
        file_path = path.relpath("/media/alexandre/Data/Documents/Alexandre2R/MOVA/TFM/video/MOT17Det/MOT17DetLabels/train/MOT17-02/gt/gt.txt")
        target = 'person'
    elif dataset == 'otb':
        file_path = path.relpath("/media/alexandre/Data/Documents/Alexandre2R/MOVA/TFM/video/OTB100/Basketball/groundtruth_rect.txt")
        target = 'person'  # hardcoded class label
    elif dataset == 'nfs':
        file_path = path.relpath("/media/alexandre/Data/Documents/Alexandre2R/MOVA/TFM/video/NFS/airboard_1.txt")
        target = 'class'
    previous_frame_number = None
    get_files(dataset, file_path, previous_frame_number, target)
