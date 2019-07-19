from os import path


# groundtruth format required: <class_name> <left> <top> <right> <bottom>
# https://github.com/rafaelpadilla/Object-Detection-Metrics#create-your-detection-files

def get_files(dataset_name, path_file):
    if dataset_name == 'mot':  # ToDo: include ids in tracking (pending); now not taking id into account
        folder_path = path_file.rsplit('/', 1)[0]
        label_classes_dict = {'1': 'person',
                              '2': 'car',
                              '3': 'car',
                              '4': 'bicycle',
                              '5': 'motorbike',
                              '6': 'bicycle',
                              '7': 'person'
                              }  # coco equivalences for MOT label classes
        with open(path_file, 'r') as stream:
            out = stream.readlines()  # <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <object_type>, <vis_ratio>
            for i in range(len(out)):
                out[i] = out[i].split(',')
                if out[i][6] != '0':  # confidence acts like a flag (entry is to be considered (1) or ignored (0))
                    with open(folder_path + '/' + str(int(out[i][0]) - 1) + '.txt',
                              'a') as logfile:  # append mode, remember to delete previous files
                        if str(out[i][7]) in label_classes_dict:
                            label = label_classes_dict[str(out[i][7])]
                        else:  # other classes are supposed to be person class
                            label = 'person'
                        logfile.write(
                            label + " " + str(out[i][2]) + " " + str(out[i][3]) + " " + str(int(
                                out[i][2]) + int(out[i][4])) + " " + str(int(out[i][3]) + int(out[i][5])) + "\n")

    # elif dataset_name == 'otb':
    #     folder_path = path_file.rsplit('/', 1)[0]
    #     with open(path_file, 'r') as stream:
    #         out = stream.readlines()  # (x, y, box-width, box-height)
    #         for i in range(len(out)):
    #             out[i] = out[i].split(',')
    #             with open(folder_path + '/' + str(i) + '.txt', 'a') as logfile:
    #                 logfile.write(
    #                     target + " " + str(out[i][0]) + " " + str(out[i][1]) + " " + str(int(
    #                         out[i][0]) + int(out[i][2])) + " " + str(int(out[i][1]) + int(out[i][3][:2])) + "\n")
    #
    # elif dataset_name == 'nfs':
    #     folder_path = path_file.rsplit('/', 1)[0]
    #     with open(path_file, 'r') as stream:
    #         out = stream.readlines()  # (*, left, top, right, bottom, frame, *, *, *, class)
    #         for i in range(len(out)):
    #             out[i] = out[i].split(' ')
    #             with open(folder_path + '/' + str(i) + '.txt', 'w') as logfile:
    #                 logfile.write(
    #                     target + " " + str(out[i][1]) + " " + str(out[i][2]) + " " + str(int(out[i][3])) + " " + str(
    #                         int(out[i][4])) + "\n")


if __name__ == "__main__":
    file_path = None
    dataset = 'mot'
    if dataset == 'mot':
        file_path = path.relpath(
            "/media/alexandre/Data/Documents/Alexandre2R/MOVA/TFM/video/MOT17Det/MOT17Det/train/MOT17-13/gt/gt.txt")
        #ToDo: move gt.txt from gt folder automatically, input file by args
    # elif dataset == 'otb':
    #     file_path = path.relpath(
    #         "/media/alexandre/Data/Documents/Alexandre2R/MOVA/TFM/video/OTB100/Basketball/groundtruth_rect.txt")
    #     target = 'person'  # hardcoded class label
    # elif dataset == 'nfs':
    #     file_path = path.relpath("/media/alexandre/Data/Documents/Alexandre2R/MOVA/TFM/video/NFS/airboard_1.txt")
    #     target = 'class'
    get_files(dataset, file_path)
