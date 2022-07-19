BUFFER=100


python demo_video.py --video=/media/Data/projects/DROID-SLAM/data/yt_gopro_mtb/NH_Youtube/NHYoutube.mp4 \
    --calib=/media/Data/projects/DROID-SLAM/calib/gopro9_wide.txt \
    --reconstruction_path=/media/Data/projects/DROID-SLAM/data/yt_gopro_mtb/NH_Youtube/NH_results --buffer=${BUFFER} --skip_seconds=5
    
python demo_video.py --video=/media/Data/projects/DROID-SLAM/data/yt_gopro_mtb/NH_Youtube/MNYoutube.mp4 \
    --calib=/media/Data/projects/DROID-SLAM/calib/gopro9_wide.txt \
    --reconstruction_path=/media/Data/projects/DROID-SLAM/data/yt_gopro_mtb/NH_Youtube/MN_results --buffer=${BUFFER} --skip_seconds=1
