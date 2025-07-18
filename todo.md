# Plan: Integrate Video Recording with WandB

## Problem Analysis
Currently, the project has video recording functionality during training (via `gym.wrappers.RecordVideo`), but the videos are only saved locally. The `Writer` class has a `log_videos` method that uploads videos to wandb, but it's only called during checkpoint saving. We need to integrate continuous video recording and uploading to wandb during training.

## Todo Items

- [x] 1. Understand current video recording setup
  - The training script uses `gym.wrappers.RecordVideo` when `--video` flag is set
  - Videos are saved locally to `writer.video_dir`
  - The `Writer.log_videos()` method exists but is only called during checkpoints

- [x] 2. Modify the Trainer class to periodically upload videos
  - Add video upload logic to the training loop
  - Call `writer.log_videos()` at regular intervals (not just during checkpoints)

- [x] 3. Ensure video recording works with the mimic task
  - Verify that `render_mode="rgb_array"` is properly passed to the environment
  - Check if the mimic environment supports rendering

- [x] 4. Add configuration options for video upload frequency
  - Add a parameter to control how often videos are uploaded to wandb
  - Make it independent from checkpoint saving frequency

- [x] 5. Test the integration
  - Run training with video recording enabled
  - Verify videos appear in wandb dashboard

## Implementation Approach
We'll make minimal changes to integrate video uploading into the existing training loop. The main changes will be:
1. Add a video upload interval parameter
2. Track when to upload videos in the Trainer
3. Call the existing `writer.log_videos()` method at appropriate intervals

## Review Summary

### Changes Made:
1. **Configuration Changes (prop_mimic.yaml)**:
   - Enabled video recording by setting `upload_videos: 1`
   - Added new parameter `video_upload_interval: 50` to control upload frequency (every 50 evaluation episodes)

2. **Trainer Class Changes (trainer.py)**:
   - Added tracking for evaluation episode count (`self.eval_episode_count`)
   - Added video upload interval configuration reading from writer
   - Implemented periodic video upload logic that:
     - Uploads videos every N evaluation episodes (configurable)
     - Works independently from checkpoint saving
     - Prints confirmation when videos are uploaded

### How It Works:
- Videos are recorded continuously during training (when enabled)
- Every 50 evaluation episodes (configurable), videos are uploaded to wandb
- Videos are also still uploaded during checkpoint saves (existing behavior)
- The feature is completely optional and controlled by configuration

### No Breaking Changes:
- All existing functionality remains unchanged
- The feature is disabled by default if `video_upload_interval` is not specified or is 0
- The implementation uses existing infrastructure (Writer.log_videos method)