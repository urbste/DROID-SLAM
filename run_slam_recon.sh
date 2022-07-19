BUFFER=1000
PATH_TO_VIDEOS=/home/zosurban/Projects/DROID-SLAM-urbste/data/steffen
CALIB=/home/zosurban/Projects/DROID-SLAM-urbste/calib/gopro9_linear.txt

python demo_video.py --video=${PATH_TO_VIDEOS}/bike1_trail1_linear.MP4 \
    --reconstruction_path=${PATH_TO_VIDEOS}/bike1_trail1_results --buffer=${BUFFER} --t0=67.0 --tend=87.0 \
    --calib=${CALIB}
    
python demo_video.py --video=${PATH_TO_VIDEOS}/bike2_trail1_linear.MP4 \
    --reconstruction_path=${PATH_TO_VIDEOS}/bike2_trail1_results --buffer=${BUFFER} --t0=30.0 --tend=50.0 \
    --calib=${CALIB}
