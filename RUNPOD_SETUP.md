FULLY AUTOMATED RUNPOD SETUP

ZERO MANUAL WORK - EVERYTHING AUTOMATED

1. Upload project folder to /workspace/

2. Run ONE command:
   cd /workspace
   bash runpod_setup.sh

3. Start training:
   cd /workspace/src/experiments
   python run.py --config configs/wilddeepfake.yaml --output /workspace/outputs

AUTOMATED PROCESS:
✓ Auto-installs all dependencies
✓ Auto-downloads WildDeepfake dataset
✓ Auto-extracts all archives
✓ Auto-preprocesses frames from videos
✓ Auto-detects FaceForensics++ (if ff-c23.zip in /workspace/)
✓ Auto-deletes zip files after extraction
✓ Auto-removes video files after frame extraction
✓ Auto-cleans HuggingFace download cache
✓ Auto-cleans temporary files

SPACE OPTIMIZATION:
- Videos deleted immediately after frame extraction
- Zip files deleted after extraction
- Temp files cleaned automatically
- Only processed images kept

OPTIONAL FACEFORENSICS++:
Upload ff-c23.zip to /workspace/ before setup
Script handles everything automatically

DATA LOCATIONS:
/workspace/data/wilddeepfake/
/workspace/data/ff_c23/
/workspace/data/torch_cache/
/workspace/data/hf_cache/

MONITOR:
nvidia-smi
tail -f /workspace/outputs/*/train.log
