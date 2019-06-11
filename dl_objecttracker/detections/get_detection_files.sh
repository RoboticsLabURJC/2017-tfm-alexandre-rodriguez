DIR="dets/"
if [ "$(ls -A $DIR)" ]; then
     echo "$DIR is not empty"
     cd dets/
     sudo rm -r *.txt && cd ..
     python2 convert_detections_to_pascalvoc.py
     python2 create_empty_files.py
else
    echo "$DIR is empty"
    python2 convert_detections_to_pascalvoc.py
    python2 create_empty_files.py
fi

