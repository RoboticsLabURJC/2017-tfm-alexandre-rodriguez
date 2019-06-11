import yaml
from os import path

# detections format required: <class_name> <confidence> <left> <top> <right> <bottom>


def get_files(path_file, previous_frame):
    with open(path_file, 'r') as stream:
        out = yaml.load(stream)  # <frame> <class_name> <confidence> <left> <top> <right> <bottom>
        for i in range(len(out)):
            actual_frame = out[i][0]
            if actual_frame == previous_frame or previous_frame is None:
                with open('dets/'+str(actual_frame) + '.txt', 'a') as logfile:
                    if len(out[i]) > 1:  # this was not an empty frame in terms of detections
                        logfile.write(
                            out[i][1] + " " + str(out[i][2]) + " " + str(out[i][3][0]) + " " + str(out[i][3][1]) + " " + str(out[i][4][0]) + " " + str(out[i][4][
                                1]) + "\n")
            elif actual_frame != previous_frame:
                with open('dets/'+str(actual_frame) + '.txt', 'a') as logfile:
                    if len(out[i]) > 1:  # this was not an empty frame in terms of detections
                        logfile.write(
                            out[i][1] + " " + str(out[i][2]) + " " + str(out[i][3][0]) + " " + str(out[i][3][1]) + " " + str(
                                out[i][4][0]) + " " + str(out[i][4][
                                                              1]) + "\n")
            previous_frame = actual_frame


if __name__ == "__main__":
    file_path = path.relpath("../log_tracking.yaml")
    previous_frame_number = None
    get_files(file_path, previous_frame_number)
    file_path = path.relpath("../log_network.yaml")
    get_files(file_path, previous_frame_number)
