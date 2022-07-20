BUFFER=500
PATH_TO_VIDEOS=/media/Data/Sparsenet/OrbSlam3/TestMappingRelocalization/JenzigTrailsJune
CALIB=/media/Data/projects/DROID-SLAM/calib/gopro9_linear.txt

RESULT1=${PATH_TO_VIDEOS}/bike1_trail1_results
RESULT2=${PATH_TO_VIDEOS}/bike2_trail1_results
MERGED=${PATH_TO_VIDEOS}/merged

python demo_video.py --video=${PATH_TO_VIDEOS}/bike1_trail1_linear.MP4 \
    --reconstruction_path=${RESULT1} --buffer=${BUFFER} --t0=67.0 --tend=80.0 \
    --calib=${CALIB} --disable_vis
    
python demo_video.py --video=${PATH_TO_VIDEOS}/bike2_trail1_linear.MP4 \
    --reconstruction_path=${RESULT2} --buffer=${BUFFER} --t0=30.0 --tend=43.0 \
    --calib=${CALIB} --disable_vis

python align_droidslam_videos.py --path1=${RESULT1} --path2=${RESULT2}
python trajectory_merger.py --path1=${RESULT1} --path2=${RESULT2} --merged=${MERGED} --disable_vis
